from __future__ import annotations

import torch

from hunch.model import Example
from hunch.transformer import (
    ByteDecoderTransformer,
    ByteTokenizer,
    ModelConfig,
)


def test_byte_tokenizer_maps_bytes_and_shifts_target_labels() -> None:
    encoded = ByteTokenizer().encode_example(
        Example(("a", "b", "c"), "déf"), block_size=64
    )

    assert encoded.input_ids == (97, 256, 98, 256, 99, 256, 100, 195, 169, 102)
    assert encoded.labels == (
        -100,
        -100,
        -100,
        -100,
        -100,
        100,
        195,
        169,
        102,
        256,
    )
    assert encoded.target_byte_count == 4


def test_default_model_configuration_is_the_confirmed_architecture() -> None:
    assert ModelConfig().as_dict() == {
        "block_size": 256,
        "decoder_blocks": 4,
        "attention_heads": 4,
        "embedding_dim": 128,
        "feed_forward_dim": 512,
        "dropout": 0.1,
    }


def test_transformer_returns_logits_and_a_target_only_loss() -> None:
    config = ModelConfig(
        block_size=64,
        decoder_blocks=1,
        attention_heads=2,
        embedding_dim=32,
        feed_forward_dim=64,
    )
    model = ByteDecoderTransformer(config)
    encoded = ByteTokenizer().encode_example(
        Example(("one", "two", "three"), "four"), config.block_size
    )
    input_ids = torch.tensor([encoded.input_ids])
    labels = torch.tensor([encoded.labels])

    logits = model(input_ids)
    loss = model.loss(input_ids, labels)

    assert logits.shape == (1, len(encoded.input_ids), 258)
    assert torch.isfinite(loss)
