from __future__ import annotations

from typing import Sequence

import torch
import pytest

from hunch.model import Example
import hunch.transformer as transformer
from hunch.transformer import (
    ByteDecoderTransformer,
    ByteTokenizer,
    ModelConfig,
    TrainingConfig,
    train_transformer,
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


def test_long_targets_use_windows_that_score_every_target_byte() -> None:
    windows = ByteTokenizer().encode_windows(
        Example(("one", "two", "three"), "a" * 256), block_size=64
    )

    assert sum(window.target_byte_count for window in windows) == 256
    assert sum(
        sum(0 <= label < 256 for label in window.labels) for window in windows
    ) == 256
    assert all(len(window.input_ids) <= 64 for window in windows)


def test_training_uses_bounded_minibatches(monkeypatch: pytest.MonkeyPatch) -> None:
    original_tensor_batch = transformer._tensor_batch
    observed_batch_sizes: list[int] = []

    def record_batch(
        examples: Sequence[Example],
        tokenizer: ByteTokenizer,
        block_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed_batch_sizes.append(len(examples))
        return original_tensor_batch(examples, tokenizer, block_size, device)

    monkeypatch.setattr(transformer, "_tensor_batch", record_batch)
    examples = [
        Example(("one", "two", "three"), f"target-{index}")
        for index in range(7)
    ]

    train_transformer(
        examples,
        examples[:1],
        TrainingConfig.tiny(epochs=1, batch_size=2),
    )

    assert observed_batch_sizes[:4] == [2, 2, 2, 1]
