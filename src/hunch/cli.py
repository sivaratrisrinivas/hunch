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
from hunch.state import StateError, load_model, save_model
from hunch.transformer import (
    EvaluationMetrics,
    TrainingConfig,
    evaluate_transformer,
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
    save_model(fit.model, state_directory())
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


def _valid_suggestion(suggestion: str) -> bool:
    if not suggestion.strip() or len(suggestion.encode("utf-8")) > MAX_SUGGESTION_BYTES:
        return False
    if "\n" in suggestion or "\r" in suggestion or is_sensitive(suggestion):
        return False
    return not any(unicodedata.category(character) == "Cc" for character in suggestion)


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
    subcommands.add_parser("predict", help="print one predicted command")
    return command_parser


def run(arguments: Sequence[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    try:
        if options.command == "train":
            return train(options)
        return predict()
    except (HistoryError, StateError, OSError, RuntimeError, ValueError) as error:
        print(f"hunch: error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("hunch: error: training interrupted", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run())
