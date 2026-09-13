from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Sequence

from hunch.model import CountModel, Example, MAX_ORDER
from hunch.state import StateError
from hunch.transformer import ByteDecoderTransformer, bits_per_byte


INSPECT_FILENAME = "inspect.jsonl"


@dataclass(frozen=True)
class FirstByte:
    token: int
    probability: float

    @property
    def text(self) -> str:
        if 33 <= self.token <= 126:
            return chr(self.token)
        return f"\\x{self.token:02x}"

    def as_dict(self) -> dict[str, float | str]:
        return {"byte": self.text, "probability": self.probability}


@dataclass(frozen=True)
class Inspection:
    context: tuple[str, ...]
    suggestion: str
    command_ngram: str
    first_bytes: tuple[FirstByte, ...]
    bits_per_byte: float

    def as_dict(self) -> dict[str, object]:
        return {
            "bits_per_byte": self.bits_per_byte,
            "command_ngram": self.command_ngram,
            "context": list(self.context),
            "first_bytes": [item.as_dict() for item in self.first_bytes],
            "suggestion": self.suggestion,
        }


def inspect_current(
    model: ByteDecoderTransformer,
    commands: Sequence[str],
    suggestion: str,
) -> Inspection:
    context = tuple(commands[-MAX_ORDER:])
    return Inspection(
        context=context,
        suggestion=suggestion,
        command_ngram=CountModel.train(commands).predict(context),
        first_bytes=tuple(
            FirstByte(token, probability)
            for token, probability in model.first_byte_top(context)
        ),
        bits_per_byte=bits_per_byte(
            model, [Example(context, suggestion)], batch_size=1
        ),
    )


def render_inspection(inspection: Inspection) -> str:
    lines = [
        inspection.suggestion,
        f"command-ngram: {inspection.command_ngram}",
    ]
    for item in inspection.first_bytes:
        lines.append(f"{item.text} {item.probability:.4f}")
    lines.append(f"bits per byte: {inspection.bits_per_byte:.4f}")
    return "\n".join(lines) + "\n"


def append_inspection(state_directory: Path, inspection: Inspection) -> Path:
    try:
        state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        state_directory.chmod(0o700)
        path = state_directory / INSPECT_FILENAME
        payload = json.dumps(
            inspection.as_dict(), ensure_ascii=False, sort_keys=True
        )
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, f"{payload}\n".encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(path, 0o600)
        return path
    except OSError as error:
        raise StateError(f"cannot write Inspection record: {error}") from error
