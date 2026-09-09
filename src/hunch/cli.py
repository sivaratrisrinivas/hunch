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
)
from hunch.state import StateError, load_model, save_model


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


def train() -> int:
    prepared = read_usable_history(history_path())
    try:
        split = chronological_split(prepared.commands)
    except ValueError as error:
        raise HistoryError(str(error)) from error

    model = CountModel.train(split.train_commands)
    validation_common = evaluate_constant(model.most_common, split.validation)
    validation_ngram = evaluate_model(model, split.validation)
    test_common = evaluate_constant(model.most_common, split.test)
    test_ngram = evaluate_model(model, split.test)
    save_model(model, state_directory())

    print(f"usable commands: {len(prepared.commands)}")
    print(f"filtered sensitive commands: {prepared.sensitive_count}")
    print(
        f"split: train={len(split.train_commands)} validation={len(split.validation)} "
        f"test={len(split.test)}"
    )
    _print_accuracy("validation most-common", validation_common)
    _print_accuracy("validation command-ngram", validation_ngram)
    _print_accuracy("test most-common", test_common)
    _print_accuracy("test command-ngram", test_ngram)
    return 0


def predict() -> int:
    prepared = read_usable_history(history_path())
    if len(prepared.commands) < MAX_ORDER:
        raise HistoryError(
            f"history is too small: need at least {MAX_ORDER} usable commands to predict"
        )
    model = load_model(state_directory())
    suggestion = model.predict(prepared.commands[-MAX_ORDER:])
    if _valid_suggestion(suggestion):
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


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="hunch")
    subcommands = command_parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("train", help="train the count model from Bash history")
    subcommands.add_parser("predict", help="print one predicted command")
    return command_parser


def run(arguments: Sequence[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    try:
        if options.command == "train":
            return train()
        return predict()
    except (HistoryError, StateError) as error:
        print(f"hunch: error: {error}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run())
