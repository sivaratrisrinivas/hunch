from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from hunch.model import Example, MAX_ORDER
from hunch.state import StateError


SCOREBOARD_FILENAME = "scoreboard.json"


def save_scoreboard(state_directory: Path, examples: Sequence[Example]) -> Path:
    try:
        state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        state_directory.chmod(0o700)
    except OSError as error:
        raise StateError(f"cannot prepare private state directory: {error}") from error

    destination = state_directory / SCOREBOARD_FILENAME
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{SCOREBOARD_FILENAME}.", suffix=".tmp", dir=state_directory
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        os.chmod(temporary_path, 0o600)
        payload = json.dumps(
            [
                {"context": list(example.context), "next": example.target}
                for example in examples
            ],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
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
        raise StateError(f"cannot save scoreboard: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def load_scoreboard(state_directory: Path) -> list[Example]:
    path = state_directory / SCOREBOARD_FILENAME
    if not path.is_file():
        raise StateError(f"scoreboard does not exist: {path}; run 'hunch setup' first")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StateError(f"scoreboard is unreadable: {path}: {error}") from error
    if not isinstance(raw, list) or not raw:
        raise StateError(f"scoreboard is unreadable: {path}: pairs required")
    examples: list[Example] = []
    for item in raw:
        if not isinstance(item, dict):
            raise StateError(f"scoreboard is unreadable: {path}: pair required")
        context = item.get("context")
        target = item.get("next")
        if (
            not isinstance(context, list)
            or len(context) != MAX_ORDER
            or not all(isinstance(command, str) for command in context)
            or not isinstance(target, str)
        ):
            raise StateError(f"scoreboard is unreadable: {path}: invalid pair")
        examples.append(Example(tuple(context), target))
    return examples
