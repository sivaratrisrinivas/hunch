from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from hunch.model import CountModel, MAX_ORDER


STATE_VERSION = 1
STATE_FILENAME = "count-model.json"


class StateError(Exception):
    """Raised when count-model state is missing or invalid."""


def save_model(model: CountModel, state_directory: Path) -> Path:
    try:
        state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        state_directory.chmod(0o700)
    except OSError as error:
        raise StateError(f"cannot prepare private state directory: {error}") from error

    path = state_directory / STATE_FILENAME
    temporary_path = state_directory / f".{STATE_FILENAME}.tmp"
    payload = {
        "version": STATE_VERSION,
        "order": MAX_ORDER,
        "most_common": model.most_common,
        "counts": {
            str(order): [
                {"context": list(context), "targets": target_counts}
                for context, target_counts in contexts.items()
            ]
            for order, contexts in model.counts.items()
        },
    }
    try:
        with temporary_path.open("w", encoding="utf-8") as state_file:
            os.chmod(temporary_path, 0o600)
            json.dump(payload, state_file, ensure_ascii=False, separators=(",", ":"))
            state_file.write("\n")
            state_file.flush()
            os.fsync(state_file.fileno())
        temporary_path.replace(path)
    except (OSError, TypeError) as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise StateError(f"cannot save model state: {error}") from error
    return path


def load_model(state_directory: Path) -> CountModel:
    path = state_directory / STATE_FILENAME
    if not path.is_file():
        raise StateError(f"model state does not exist: {path}; run 'hunch train' first")
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != STATE_VERSION:
            raise ValueError("unsupported state version")
        most_common = payload["most_common"]
        serialized_counts = payload["counts"]
        if not isinstance(most_common, str) or not isinstance(serialized_counts, dict):
            raise ValueError("invalid state fields")
        counts: dict[int, dict[tuple[str, ...], dict[str, int]]] = {}
        for order in range(1, MAX_ORDER + 1):
            entries = serialized_counts[str(order)]
            contexts: dict[tuple[str, ...], dict[str, int]] = {}
            for entry in entries:
                context = tuple(_string_list(entry["context"]))
                targets = entry["targets"]
                if len(context) != order or not isinstance(targets, dict):
                    raise ValueError("invalid count entry")
                contexts[context] = {
                    target: count
                    for target, count in targets.items()
                    if isinstance(target, str) and isinstance(count, int) and count > 0
                }
                if not contexts[context]:
                    raise ValueError("empty target counts")
            counts[order] = contexts
        return CountModel(most_common, counts)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise StateError(f"model state is unreadable: {path}: {error}") from error


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("expected a list of commands")
    return value
