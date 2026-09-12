from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from hunch.history import HistoryError
from hunch.model import (
    Accuracy,
    CountModel,
    Example,
    MAX_ORDER,
    evaluate_model,
    examples_from_commands,
)
from hunch.pile import PILE_SIZE, load_consumed, save_consumed
from hunch.scoreboard import load_scoreboard
from hunch.state import STATE_FILENAME, StateError, load_model, save_model
from hunch.transformer import (
    TrainingConfig,
    continue_transformer,
    evaluate_transformer,
    resolve_device,
)


UPDATE_LOG_FILENAME = "update.log"
Decision = Literal["keep", "discard"]


@dataclass(frozen=True)
class UpdateResult:
    decision: Decision
    transformer: Accuracy
    ngram: Accuracy

    @property
    def record(self) -> str:
        return (
            f"{self.decision} "
            f"transformer exact-command accuracy: {self.transformer.percent:.2f}% "
            f"({self.transformer.correct}/{self.transformer.total}) "
            f"command-ngram exact-command accuracy: {self.ngram.percent:.2f}% "
            f"({self.ngram.correct}/{self.ngram.total})"
        )


def last_update_record(state_directory: Path) -> str | None:
    path = state_directory / UPDATE_LOG_FILENAME
    if not path.is_file():
        return None
    try:
        lines = [
            line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    except (OSError, UnicodeError) as error:
        raise StateError(f"update log is unreadable: {path}: {error}") from error
    return lines[-1] if lines else None


def apply_update(
    state_directory: Path,
    commands: Sequence[str],
    config: TrainingConfig,
) -> UpdateResult:
    if not (state_directory / STATE_FILENAME).is_file():
        raise StateError("no Champion; run 'hunch setup' first")

    consumed = load_consumed(state_directory)
    if consumed is None:
        consumed = len(commands)
    if len(commands) - consumed < PILE_SIZE:
        raise HistoryError("no pile of eight new usable commands")

    scoreboard = load_scoreboard(state_directory)
    old_examples = examples_from_commands(commands[:consumed])
    new_examples = _pile_examples(commands, consumed)
    if not old_examples:
        raise HistoryError("no commands from before the Pile to mix into the Update")

    device = resolve_device(config.device)
    champion = load_model(state_directory, device)
    before = evaluate_transformer(
        champion, scoreboard, batch_size=config.batch_size
    )
    candidate = continue_transformer(champion, old_examples, new_examples, config)
    after = evaluate_transformer(
        candidate, scoreboard, batch_size=config.batch_size
    )
    decision: Decision = (
        "keep"
        if after.exact_accuracy.correct >= before.exact_accuracy.correct
        else "discard"
    )
    if decision == "keep":
        save_model(candidate, state_directory)

    ngram = evaluate_model(
        CountModel.train(commands[: consumed + PILE_SIZE]), scoreboard
    )
    result = UpdateResult(decision, after.exact_accuracy, ngram)
    save_consumed(state_directory, consumed + PILE_SIZE)
    _append_update_log(state_directory, result.record)
    return result


def _pile_examples(commands: Sequence[str], consumed: int) -> list[Example]:
    return [
        Example(tuple(commands[index - MAX_ORDER : index]), commands[index])
        for index in range(consumed, consumed + PILE_SIZE)
    ]


def _append_update_log(state_directory: Path, record: str) -> None:
    try:
        state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        state_directory.chmod(0o700)
        path = state_directory / UPDATE_LOG_FILENAME
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        try:
            os.write(descriptor, f"{record}\n".encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(path, 0o600)
    except OSError as error:
        raise StateError(f"cannot write update log: {error}") from error
