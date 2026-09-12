from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

from hunch.state import StateError


STATS_FILENAME = "stats.json"
_FIELDS = ("training_runs", "suggestions_displayed", "suggestions_inserted")
_RECORDABLE = {
    "displayed": "suggestions_displayed",
    "inserted": "suggestions_inserted",
}


@dataclass(frozen=True)
class Stats:
    training_runs: int = 0
    suggestions_displayed: int = 0
    suggestions_inserted: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "training_runs": self.training_runs,
            "suggestions_displayed": self.suggestions_displayed,
            "suggestions_inserted": self.suggestions_inserted,
        }


def load_stats(state_directory: Path) -> Stats:
    path = state_directory / STATS_FILENAME
    if not path.is_file():
        return Stats()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StateError(f"statistics are unreadable: {path}: {error}") from error
    return _from_mapping(raw, path)


def increment_stat(state_directory: Path, field: str) -> Stats:
    if field not in _FIELDS:
        raise ValueError(f"unknown statistic: {field}")
    values = load_stats(state_directory).as_dict()
    values[field] += 1
    updated = Stats(**values)
    save_stats(state_directory, updated)
    return updated


def record_event(state_directory: Path, event: str) -> Stats:
    try:
        field = _RECORDABLE[event]
    except KeyError as error:
        raise ValueError(f"unknown record event: {event}") from error
    return increment_stat(state_directory, field)


def save_stats(state_directory: Path, stats: Stats) -> Path:
    try:
        state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        state_directory.chmod(0o700)
    except OSError as error:
        raise StateError(f"cannot prepare private state directory: {error}") from error

    destination = state_directory / STATS_FILENAME
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{STATS_FILENAME}.", suffix=".tmp", dir=state_directory
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        os.chmod(temporary_path, 0o600)
        payload = json.dumps(stats.as_dict(), sort_keys=True) + "\n"
        with temporary_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        os.chmod(destination, 0o600)
        return destination
    except OSError as error:
        raise StateError(f"cannot save statistics: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def render_stats(stats: Stats, last_update: str | None = None) -> str:
    text = (
        f"training runs: {stats.training_runs}\n"
        f"suggestions displayed: {stats.suggestions_displayed}\n"
        f"suggestions inserted: {stats.suggestions_inserted}\n"
    )
    if last_update:
        return f"{text}last update: {last_update}\n"
    return text


def _from_mapping(raw: object, path: Path) -> Stats:
    if not isinstance(raw, dict):
        raise StateError(f"statistics are unreadable: {path}: mapping required")
    values: dict[str, int] = {}
    for field in _FIELDS:
        value = raw.get(field, 0)
        if type(value) is not int or isinstance(value, bool) or value < 0:
            raise StateError(f"statistics are unreadable: {path}: invalid {field}")
        values[field] = value
    return Stats(**values)
