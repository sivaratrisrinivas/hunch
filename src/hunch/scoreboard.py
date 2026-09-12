from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from hunch.model import Example
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
