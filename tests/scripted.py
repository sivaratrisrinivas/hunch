from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from hunch.state import save_model
from hunch.transformer import (
    COMMAND_BOUNDARY_TOKEN,
    ByteDecoderTransformer,
    ModelConfig,
)


def scripted_transformer(
    context: tuple[str, str, str],
    continuation: bytes,
    *,
    emit_boundary: bool = True,
) -> ByteDecoderTransformer:
    config = ModelConfig(
        block_size=64,
        decoder_blocks=1,
        attention_heads=2,
        embedding_dim=32,
        feed_forward_dim=64,
        dropout=0.0,
    )
    model = ByteDecoderTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2)
    prefix = model.tokenizer.serialize_context(context)
    target = list(continuation) + ([COMMAND_BOUNDARY_TOKEN] if emit_boundary else [])
    model.train()
    for _ in range(80):
        tokens = list(prefix)
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for token in target:
            window = tokens[-model.config.block_size :]
            input_ids = torch.tensor([window], dtype=torch.long)
            logits = model(input_ids)[0, -1]
            losses.append(
                F.cross_entropy(logits.unsqueeze(0), torch.tensor([token]))
            )
            tokens.append(token)
        torch.stack(losses).mean().backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    model.eval()
    return model


def install_scripted_checkpoint(
    home: Path,
    context: tuple[str, str, str],
    continuation: bytes,
    *,
    emit_boundary: bool = True,
) -> Path:
    model = scripted_transformer(
        context, continuation, emit_boundary=emit_boundary
    )
    return save_model(model, home / ".local" / "share" / "hunch")
