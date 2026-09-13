from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
from typing import Iterator, Literal, Sequence

from hunch.exam import Exam, save_exam
from hunch.history import HistoryError
from hunch.model import Accuracy, CountModel, evaluate_model
from hunch.pile import PILE_SIZE, UpdateExamples, load_consumed, save_consumed
from hunch.scoreboard import load_scoreboard
from hunch.state import STATE_FILENAME, StateError, load_model, save_model
from hunch.transformer import (
    TrainingConfig,
    continue_transformer,
    exact_hits,
    resolve_device,
)


UPDATE_LOG_FILENAME = "update.log"
UPDATE_LOCK_FILENAME = "update.lock"
Decision = Literal["keep", "discard", "wait"]


@dataclass(frozen=True)
class UpdateResult:
    decision: Decision
    transformer: Accuracy | None
    ngram: Accuracy

    @property
    def record(self) -> str:
        ngram = (
            f"command-ngram exact-command accuracy: {self.ngram.percent:.2f}% "
            f"({self.ngram.correct}/{self.ngram.total})"
        )
        if self.decision == "wait":
            return f"the transformer waited {ngram}"
        assert self.transformer is not None
        return (
            f"{self.decision} "
            f"transformer exact-command accuracy: {self.transformer.percent:.2f}% "
            f"({self.transformer.correct}/{self.transformer.total}) "
            f"{ngram}"
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

    with _exclusive_update(state_directory):
        return _apply_update(state_directory, commands, config)


def _apply_update(
    state_directory: Path,
    commands: Sequence[str],
    config: TrainingConfig,
) -> UpdateResult:
    consumed = load_consumed(state_directory)
    if consumed is None:
        consumed = len(commands)
    scoreboard = load_scoreboard(state_directory)
    batch = UpdateExamples.from_stream(commands, consumed, scoreboard)
    decision: Decision
    transformer: Accuracy | None
    if _transformer_should_wait(config):
        decision = "wait"
        transformer = None
        exam = Exam.waited_out()
    else:
        if not batch.old:
            raise HistoryError(
                "no commands from before the Pile to mix into the Update"
            )

        device = resolve_device(config.device)
        champion = load_model(state_directory, device)
        before = exact_hits(champion, scoreboard)
        candidate = continue_transformer(champion, batch, config)
        after = exact_hits(candidate, scoreboard)
        decision = "keep" if sum(after) >= sum(before) else "discard"
        if decision == "keep":
            save_model(candidate, state_directory)
        transformer = Accuracy(sum(after), len(scoreboard))
        exam = Exam.from_hits(scoreboard, before, after)

    ngram = evaluate_model(
        CountModel.train(commands[: consumed + PILE_SIZE]), scoreboard
    )
    result = UpdateResult(decision, transformer, ngram)
    save_consumed(state_directory, consumed + PILE_SIZE)
    _append_update_log(state_directory, result.record)
    save_exam(state_directory, exam)
    return result


def _transformer_should_wait(config: TrainingConfig) -> bool:
    return config.device != "cpu" and resolve_device("cuda").type != "cuda"


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


@contextmanager
def _exclusive_update(state_directory: Path) -> Iterator[None]:
    try:
        state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        state_directory.chmod(0o700)
        lock_path = state_directory / UPDATE_LOCK_FILENAME
        handle = open(lock_path, "a+b")
        os.chmod(lock_path, 0o600)
    except OSError as error:
        raise StateError(f"cannot lock Update: {error}") from error
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise StateError("an Update is already running") from error
    try:
        yield
    finally:
        handle.close()
