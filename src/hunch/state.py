from __future__ import annotations

import os
from pathlib import Path
import pickle
import tempfile
from typing import Mapping

import torch

from hunch.transformer import (
    ARCHITECTURE_VERSION,
    ByteDecoderTransformer,
    ModelConfig,
    TOKENIZER_VERSION,
)


STATE_VERSION = 1
STATE_FILENAME = "transformer.pt"


class StateError(Exception):
    """Raised when transformer state is missing, invalid, or cannot be saved."""


def save_model(model: ByteDecoderTransformer, state_directory: Path) -> Path:
    """Publish one complete, private transformer checkpoint atomically."""
    try:
        state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        state_directory.chmod(0o700)
    except OSError as error:
        raise StateError(f"cannot prepare private state directory: {error}") from error

    destination = state_directory / STATE_FILENAME
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{STATE_FILENAME}.", suffix=".tmp", dir=state_directory
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        os.chmod(temporary_path, 0o600)
        payload = _checkpoint_payload(model)
        with temporary_path.open("wb") as checkpoint:
            torch.save(payload, checkpoint)
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
        _load_checkpoint_path(temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
        _best_effort_sync_directory(state_directory)
        return destination
    except (
        OSError,
        pickle.PicklingError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        raise StateError(f"cannot save transformer state: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def load_model(
    state_directory: Path, device: torch.device | None = None
) -> ByteDecoderTransformer:
    destination = state_directory / STATE_FILENAME
    if not destination.is_file():
        raise StateError(
            f"model state does not exist: {destination}; run 'hunch train' first"
        )
    model = _read_checkpoint(destination)
    target = device or torch.device("cpu")
    if target.type == "cpu":
        return model
    try:
        return model.to(target)
    except (RuntimeError, OSError):
        return _read_checkpoint(destination)


def _checkpoint_payload(model: ByteDecoderTransformer) -> dict[str, object]:
    return {
        "format_version": STATE_VERSION,
        "architecture": ARCHITECTURE_VERSION,
        "tokenizer": {
            "version": TOKENIZER_VERSION,
            "byte_ids": "identity-0-255",
            "command_boundary_id": model.tokenizer.boundary_token,
            "padding_id": model.tokenizer.padding_token,
        },
        "model_config": model.config.as_dict(),
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
    }


def _read_checkpoint(path: Path) -> ByteDecoderTransformer:
    try:
        return _load_checkpoint_path(path)
    except (
        OSError,
        EOFError,
        IndexError,
        KeyError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise StateError(f"model state is unreadable: {path}: {error}") from error


def _load_checkpoint_path(path: Path) -> ByteDecoderTransformer:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    if payload.get("format_version") != STATE_VERSION:
        raise ValueError("unsupported checkpoint version")
    if payload.get("architecture") != ARCHITECTURE_VERSION:
        raise ValueError("unsupported checkpoint architecture")

    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, dict) or tokenizer != {
        "version": TOKENIZER_VERSION,
        "byte_ids": "identity-0-255",
        "command_boundary_id": 256,
        "padding_id": 257,
    }:
        raise ValueError("invalid tokenizer metadata")

    raw_config = payload.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("invalid model configuration")
    config = ModelConfig.from_mapping(raw_config)
    raw_state = payload.get("state_dict")
    if not isinstance(raw_state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in raw_state.items()
    ):
        raise ValueError("invalid model state dictionary")

    model = ByteDecoderTransformer(config)
    model.load_state_dict(dict(raw_state), strict=True)
    return model


def _best_effort_sync_directory(state_directory: Path) -> None:
    try:
        descriptor = os.open(state_directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass
