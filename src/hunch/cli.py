from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence
import unicodedata

from hunch.history import HistoryError, is_sensitive, read_usable_history
from hunch.model import (
    Accuracy,
    CountModel,
    MAX_ORDER,
    chronological_split,
    evaluate_constant,
    evaluate_model,
    examples_from_commands,
)
from hunch.scoreboard import save_scoreboard
from hunch.shell import BASH_INTEGRATION, install_hook
from hunch.state import StateError, load_model, save_model
from hunch.stats import increment_stat, load_stats, record_event, render_stats
from hunch.transformer import (
    EvaluationMetrics,
    TrainingConfig,
    evaluate_transformer,
    require_cuda_device,
    resolve_device,
    train_transformer,
)


MAX_SUGGESTION_BYTES = 256


def history_path() -> Path:
    configured = os.environ.get("HUNCH_HISTORY_PATH") or os.environ.get("HISTFILE")
    return Path(configured) if configured else Path.home() / ".bash_history"


def state_directory() -> Path:
    configured = os.environ.get("HUNCH_STATE_DIR")
    if configured:
        return Path(configured)
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return root / "hunch"


def setup(options: argparse.Namespace) -> int:
    require_cuda_device()
    options.device = "cuda"
    trained = train(options)
    if trained != 0:
        return trained
    install_hook(Path.home() / ".bash_aliases")
    print(
        "Next prompt can show the first suggestion. "
        "Open a new terminal, or source ~/.bash_aliases and press Enter."
    )
    return 0


def train(options: argparse.Namespace) -> int:
    prepared = read_usable_history(history_path())
    try:
        split = chronological_split(prepared.commands)
    except ValueError as error:
        raise HistoryError(str(error)) from error

    count_model = CountModel.train(split.train_commands)
    validation_common = evaluate_constant(count_model.most_common, split.validation)
    validation_ngram = evaluate_model(count_model, split.validation)
    test_common = evaluate_constant(count_model.most_common, split.test)
    test_ngram = evaluate_model(count_model, split.test)
    training_config = _training_config(options)
    fit = train_transformer(
        examples_from_commands(split.train_commands),
        split.validation,
        training_config,
    )
    validation_transformer = evaluate_transformer(
        fit.model, split.validation, batch_size=training_config.batch_size
    )
    test_transformer = evaluate_transformer(
        fit.model, split.test, batch_size=training_config.batch_size
    )
    print(f"usable commands: {len(prepared.commands)}")
    print(f"filtered sensitive commands: {prepared.sensitive_count}")
    print(
        f"split: train={len(split.train_commands)} validation={len(split.validation)} "
        f"test={len(split.test)}"
    )
    print(f"device: {fit.device}")
    print(f"best validation loss: {fit.best_validation_loss:.6f}")
    print(f"best epoch: {fit.best_epoch}")
    _print_metrics("validation transformer", validation_transformer)
    _print_metrics("test transformer", test_transformer)
    _print_accuracy("validation most-common", validation_common)
    _print_accuracy("validation command-ngram", validation_ngram)
    _print_accuracy("test most-common", test_common)
    _print_accuracy("test command-ngram", test_ngram)
    destination = state_directory()
    save_model(fit.model, destination)
    save_scoreboard(destination, split.validation)
    increment_stat(destination, "training_runs")
    return 0


def predict() -> int:
    prepared = read_usable_history(history_path())
    if len(prepared.commands) < MAX_ORDER:
        raise HistoryError(
            f"history is too small: need at least {MAX_ORDER} usable commands to predict"
        )
    model = load_model(state_directory(), resolve_device("auto"))
    suggestion = model.generate(prepared.commands[-MAX_ORDER:])
    if suggestion is not None and _valid_suggestion(suggestion):
        print(suggestion)
    return 0


def show_stats() -> int:
    print(render_stats(load_stats(state_directory())), end="")
    return 0


def record(event: str) -> int:
    record_event(state_directory(), event)
    return 0


def shell_init() -> int:
    print(BASH_INTEGRATION, end="")
    return 0


def _valid_suggestion(suggestion: str) -> bool:
    if not suggestion.strip() or len(suggestion.encode("utf-8")) > MAX_SUGGESTION_BYTES:
        return False
    if is_sensitive(suggestion):
        return False
    return not any(
        unicodedata.category(character) in {"Cc", "Zl", "Zp"} for character in suggestion
    )


def _print_accuracy(label: str, accuracy: Accuracy) -> None:
    print(
        f"{label} exact-command accuracy: {accuracy.percent:.2f}% "
        f"({accuracy.correct}/{accuracy.total})"
    )


def _print_metrics(label: str, metrics: EvaluationMetrics) -> None:
    _print_accuracy(label, metrics.exact_accuracy)
    print(f"{label} bits per byte: {metrics.bits_per_byte:.4f}")


def _training_config(options: argparse.Namespace) -> TrainingConfig:
    defaults = TrainingConfig.tiny() if options.tiny else TrainingConfig()
    return TrainingConfig(
        model=defaults.model,
        epochs=options.epochs if options.epochs is not None else defaults.epochs,
        batch_size=(
            options.batch_size
            if options.batch_size is not None
            else defaults.batch_size
        ),
        seed=options.seed if options.seed is not None else defaults.seed,
        learning_rate=defaults.learning_rate,
        weight_decay=defaults.weight_decay,
        device=options.device if options.device is not None else defaults.device,
    )


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="hunch")
    subcommands = command_parser.add_subparsers(dest="command", required=True)
    train_parser = subcommands.add_parser(
        "train", help="train the byte-level transformer from Bash history"
    )
    train_parser.add_argument(
        "--tiny", action="store_true", help="use a small CPU profile"
    )
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default=None
    )
    setup_parser = subcommands.add_parser(
        "setup", help="train on one local GPU and install the Bash hook"
    )
    setup_parser.add_argument(
        "--tiny", action="store_true", help="use a small training profile"
    )
    setup_parser.add_argument("--epochs", type=int, default=None)
    setup_parser.add_argument("--batch-size", type=int, default=None)
    setup_parser.add_argument("--seed", type=int, default=None)
    subcommands.add_parser("predict", help="print one predicted command")
    subcommands.add_parser(
        "stats", help="show training runs and suggestion counts"
    )
    subcommands.add_parser(
        "shell-init", help="print Bash integration for prompt suggestions"
    )
    record_parser = subcommands.add_parser(
        "record", help="record a displayed or inserted suggestion"
    )
    record_parser.add_argument("event", choices=("displayed", "inserted"))
    return command_parser


def run(arguments: Sequence[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    try:
        if options.command == "train":
            return train(options)
        if options.command == "setup":
            return setup(options)
        if options.command == "predict":
            return predict()
        if options.command == "stats":
            return show_stats()
        if options.command == "shell-init":
            return shell_init()
        return record(options.event)
    except (HistoryError, StateError, OSError, RuntimeError, ValueError) as error:
        print(f"hunch: error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("hunch: error: training interrupted", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run())
