from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re


class HistoryLine(Enum):
    EMPTY = "empty"
    TIMESTAMP = "timestamp"
    SENSITIVE = "sensitive"
    USABLE = "usable"


class SensitiveKind(Enum):
    ASSIGNMENT = "ASSIGNMENT"
    FLAG = "FLAG"
    HEADER = "HEADER"
    SCHEME_TOKEN = "SCHEME_TOKEN"
    URL_USERINFO = "URL_USERINFO"
    PEM = "PEM"
    SSH_IDENTITY = "SSH_IDENTITY"


@dataclass(frozen=True)
class SensitiveRule:
    kind: SensitiveKind
    ere: str


_TIMESTAMP = re.compile(r"#\d+")

SENSITIVE_ERES: tuple[SensitiveRule, ...] = (
    SensitiveRule(
        SensitiveKind.ASSIGNMENT,
        r"(^|[^a-z0-9_])(export[[:space:]]+)?[a-z0-9_-]*"
        r"(password|passwd|passphrase|token|secret|api[_-]?key|private[_-]?key)"
        r"[a-z0-9_-]*[[:space:]]*[=:][[:space:]]*[^[:space:]]+",
    ),
    SensitiveRule(
        SensitiveKind.FLAG,
        r"(^|[[:space:]])--?(password|passwd|passphrase|token|secret|"
        r"api[_-]?key|private[_-]?key)[[:space:]]+[^[:space:]]+",
    ),
    SensitiveRule(
        SensitiveKind.HEADER,
        r"authorization[[:space:]]*:[[:space:]]*(bearer|basic)[[:space:]]+"
        r"[^[:space:]]+",
    ),
    SensitiveRule(
        SensitiveKind.SCHEME_TOKEN,
        r"(^|[^a-z0-9_])(bearer|basic)[[:space:]]+[a-z0-9._~+/=-]{8,}",
    ),
    SensitiveRule(
        SensitiveKind.URL_USERINFO,
        r"(^|[^a-z0-9_])[a-z][a-z0-9+.-]*://[^[:space:]/:@]+:[^[:space:]/@]+@",
    ),
    SensitiveRule(
        SensitiveKind.PEM,
        r"-----begin (rsa |ec |openssh )?private key-----",
    ),
    SensitiveRule(
        SensitiveKind.SSH_IDENTITY,
        r"(^|[[:space:]])ssh(-add)?[[:space:]].*(-i[[:space:]]+|/)id_"
        r"(rsa|dsa|ecdsa|ed25519)([^a-z0-9_]|$)",
    ),
)

_POSIX_TO_PYTHON = (
    ("[^[:space:]/:@]", r"[^\s/:@]"),
    ("[^[:space:]/@]", r"[^\s/@]"),
    ("[^[:space:]]", r"\S"),
    ("[[:space:]]", r"\s"),
)


def ere_to_python(ere: str) -> str:
    rewritten = ere
    for posix, python in _POSIX_TO_PYTHON:
        rewritten = rewritten.replace(posix, python)
    if "[:" in rewritten:
        raise ValueError(f"unsupported POSIX class in ERE: {ere}")
    return rewritten


_SENSITIVE_PYTHON = tuple(
    re.compile(ere_to_python(rule.ere)) for rule in SENSITIVE_ERES
)


def _matches_sensitive(physical: str) -> bool:
    lowered = physical.lower()
    return any(pattern.search(lowered) for pattern in _SENSITIVE_PYTHON)


def classify_history_line(line: str) -> HistoryLine:
    physical = line.removesuffix("\r")
    trimmed = physical.strip()
    if not trimmed:
        return HistoryLine.EMPTY
    if _TIMESTAMP.fullmatch(trimmed):
        return HistoryLine.TIMESTAMP
    if _matches_sensitive(physical):
        return HistoryLine.SENSITIVE
    return HistoryLine.USABLE


def is_usable(physical_line: str) -> bool:
    return classify_history_line(physical_line) is HistoryLine.USABLE


def is_sensitive(command: str) -> bool:
    return classify_history_line(command) is HistoryLine.SENSITIVE


def awk_count_usable() -> str:
    skips = "\n".join(
        f"  if (lower ~ /{_awk_escape_ere(rule.ere)}/) next" for rule in SENSITIVE_ERES
    )
    return (
        "{\n"
        "  line = $0\n"
        '  sub(/\\r$/, "", line)\n'
        "  trimmed = line\n"
        '  gsub(/^[[:space:]]+|[[:space:]]+$/, "", trimmed)\n'
        '  if (trimmed == "" || trimmed ~ /^#[0-9]+$/) next\n'
        "  lower = tolower(line)\n"
        f"{skips}\n"
        "  n++\n"
        "}\n"
        "END { print n + 0 }\n"
    )


def _awk_escape_ere(ere: str) -> str:
    return ere.replace("\\", "\\\\").replace("/", "\\/")


class HistoryError(Exception):
    """Raised when Bash history cannot safely be prepared."""


@dataclass(frozen=True)
class PreparedHistory:
    commands: list[str]
    sensitive_count: int


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
        kind = classify_history_line(command)
        if kind is HistoryLine.SENSITIVE:
            sensitive_count += 1
            continue
        if kind is HistoryLine.USABLE:
            commands.append(command)

    if not commands:
        raise HistoryError("history has no usable commands")
    return PreparedHistory(commands, sensitive_count)
