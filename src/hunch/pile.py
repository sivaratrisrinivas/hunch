from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from hunch.state import StateError


PILE_SIZE = 8
CONSUMED_FILENAME = "consumed.json"


def load_consumed(state_directory: Path) -> int | None:
    path = state_directory / CONSUMED_FILENAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StateError(f"pile watermark is unreadable: {path}: {error}") from error
    if not isinstance(raw, dict):
        raise StateError(f"pile watermark is unreadable: {path}: mapping required")
    consumed = raw.get("consumed")
    if type(consumed) is not int or isinstance(consumed, bool) or consumed < 0:
        raise StateError(f"pile watermark is unreadable: {path}: invalid consumed")
    return consumed


def save_consumed(state_directory: Path, consumed: int) -> Path:
    if type(consumed) is not int or consumed < 0:
        raise ValueError("consumed must be a non-negative integer")
    try:
        state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        state_directory.chmod(0o700)
    except OSError as error:
        raise StateError(f"cannot prepare private state directory: {error}") from error

    destination = state_directory / CONSUMED_FILENAME
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{CONSUMED_FILENAME}.", suffix=".tmp", dir=state_directory
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        os.chmod(temporary_path, 0o600)
        payload = json.dumps({"consumed": consumed}, sort_keys=True) + "\n"
        with temporary_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        os.chmod(destination, 0o600)
        return destination
    except OSError as error:
        raise StateError(f"cannot save pile watermark: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
