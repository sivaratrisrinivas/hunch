from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_TIMESTAMP = re.compile(r"#\d+")
_SENSITIVE_PATTERNS = (
    re.compile(
        r"(?i)(?<![a-z0-9_])(?:export\s+)?[a-z0-9_-]*"
        r"(?:password|passwd|passphrase|token|secret|api[_-]?key|private[_-]?key)"
        r"[a-z0-9_-]*\s*(?:=|:)\s*[^\s]+"
    ),
    re.compile(
        r"(?i)(?:^|\s)--?(?:password|passwd|passphrase|token|secret|"
        r"api[_-]?key|private[_-]?key)\s+\S+"
    ),
    re.compile(r"(?i)authorization\s*:\s*(?:bearer|basic)\s+\S+"),
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:^|\s)ssh(?:-add)?\s+.*(?:-i\s+|/)id_(?:rsa|dsa|ecdsa|ed25519)\b"),
)


class HistoryError(Exception):
    """Raised when Bash history cannot safely be prepared."""


@dataclass(frozen=True)
class PreparedHistory:
    commands: list[str]
    sensitive_count: int


def is_sensitive(command: str) -> bool:
    return any(pattern.search(command) for pattern in _SENSITIVE_PATTERNS)


def read_usable_history(path: Path) -> PreparedHistory:
    if not path.is_file():
        raise HistoryError(f"history file does not exist: {path}")

    try:
        contents = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise HistoryError(f"cannot read history file: {path}: {error}") from error

    commands: list[str] = []
    sensitive_count = 0
    for physical_line in contents.split("\n"):
        command = physical_line.removesuffix("\r")
        if not command.strip() or _TIMESTAMP.fullmatch(command.strip()):
            continue
        if is_sensitive(command):
            sensitive_count += 1
            continue
        commands.append(command)

    if not commands:
        raise HistoryError("history has no usable commands")
    return PreparedHistory(commands, sensitive_count)
