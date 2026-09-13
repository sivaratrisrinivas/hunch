from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from hunch.model import Example, MAX_ORDER
from hunch.state import StateError


EXAM_FILENAME = "exam.json"


@dataclass(frozen=True)
class Exam:
    waited: bool
    lost: tuple[Example, ...]
    gained: tuple[Example, ...]

    def __post_init__(self) -> None:
        if self.waited and (self.lost or self.gained):
            raise ValueError("a wait Exam has no pairs")

    @classmethod
    def waited_out(cls) -> Exam:
        return cls(waited=True, lost=(), gained=())

    @classmethod
    def from_hits(
        cls,
        examples: Sequence[Example],
        before: Sequence[bool],
        after: Sequence[bool],
    ) -> Exam:
        lost: list[Example] = []
        gained: list[Example] = []
        for example, old, new in zip(examples, before, after, strict=True):
            if old and not new:
                lost.append(example)
            elif not old and new:
                gained.append(example)
        return cls(waited=False, lost=tuple(lost), gained=tuple(gained))

    def as_dict(self) -> dict[str, object]:
        return {
            "gained": [_pair(example) for example in self.gained],
            "lost": [_pair(example) for example in self.lost],
            "waited": self.waited,
        }


def render_exam(exam: Exam) -> str:
    if exam.waited:
        return "the transformer waited\n"
    lines = [f"lost {len(exam.lost)}", f"gained {len(exam.gained)}"]
    for example in exam.lost:
        lines.append(f"lost {_format_pair(example)}")
    for example in exam.gained:
        lines.append(f"gained {_format_pair(example)}")
    return "\n".join(lines) + "\n"


def save_exam(state_directory: Path, exam: Exam) -> Path:
    try:
        state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        state_directory.chmod(0o700)
    except OSError as error:
        raise StateError(f"cannot prepare private state directory: {error}") from error

    destination = state_directory / EXAM_FILENAME
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{EXAM_FILENAME}.", suffix=".tmp", dir=state_directory
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        os.chmod(temporary_path, 0o600)
        payload = json.dumps(
            exam.as_dict(), ensure_ascii=False, indent=2, sort_keys=True
        )
        with temporary_path.open("w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        os.chmod(destination, 0o600)
        return destination
    except OSError as error:
        raise StateError(f"cannot write Exam: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def load_exam(state_directory: Path) -> Exam:
    path = state_directory / EXAM_FILENAME
    if not path.is_file():
        raise StateError("no Exam; run 'hunch update' first")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StateError(f"Exam is unreadable: {path}: {error}") from error
    return _from_mapping(raw, path)


def _pair(example: Example) -> dict[str, object]:
    return {"context": list(example.context), "next": example.target}


def _format_pair(example: Example) -> str:
    return f"{' / '.join(example.context)} -> {example.target}"


def _from_mapping(raw: object, path: Path) -> Exam:
    if not isinstance(raw, dict):
        raise StateError(f"Exam is unreadable: {path}: mapping required")
    waited = raw.get("waited")
    if type(waited) is not bool:
        raise StateError(f"Exam is unreadable: {path}: invalid waited")
    lost = _pairs(raw.get("lost"), path, "lost")
    gained = _pairs(raw.get("gained"), path, "gained")
    if waited and (lost or gained):
        raise StateError(f"Exam is unreadable: {path}: a wait Exam has no pairs")
    return Exam(waited=waited, lost=lost, gained=gained)


def _pairs(raw: object, path: Path, field: str) -> tuple[Example, ...]:
    if not isinstance(raw, list):
        raise StateError(f"Exam is unreadable: {path}: invalid {field}")
    examples: list[Example] = []
    for item in raw:
        if not isinstance(item, dict):
            raise StateError(f"Exam is unreadable: {path}: invalid {field} pair")
        context = item.get("context")
        target = item.get("next")
        if (
            not isinstance(context, list)
            or len(context) != MAX_ORDER
            or not all(isinstance(command, str) for command in context)
            or not isinstance(target, str)
        ):
            raise StateError(f"Exam is unreadable: {path}: invalid {field} pair")
        examples.append(Example(tuple(context), target))
    return tuple(examples)
