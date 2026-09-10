"""Byte-level decoder-only transformer training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Literal, Mapping, Sequence, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hunch.model import Accuracy, Example


BYTE_VOCAB_SIZE = 256
COMMAND_BOUNDARY_TOKEN = BYTE_VOCAB_SIZE
PADDING_TOKEN = COMMAND_BOUNDARY_TOKEN + 1
VOCAB_SIZE = PADDING_TOKEN + 1
IGNORE_INDEX = -100
COMMAND_CONTEXT_SIZE = 3
MAX_GENERATED_BYTES = 256
ARCHITECTURE_VERSION = "byte-decoder-preln-v1"
TOKENIZER_VERSION = 1
DevicePreference = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class ModelConfig:
    block_size: int = 256
    decoder_blocks: int = 4
    attention_heads: int = 4
    embedding_dim: int = 128
    feed_forward_dim: int = 512
    dropout: float = 0.1

    def validate(self) -> None:
        integer_fields = (
            ("block_size", self.block_size),
            ("decoder_blocks", self.decoder_blocks),
            ("attention_heads", self.attention_heads),
            ("embedding_dim", self.embedding_dim),
            ("feed_forward_dim", self.feed_forward_dim),
        )
        for name, value in integer_fields:
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embedding_dim % self.attention_heads != 0:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "block_size": self.block_size,
            "decoder_blocks": self.decoder_blocks,
            "attention_heads": self.attention_heads,
            "embedding_dim": self.embedding_dim,
            "feed_forward_dim": self.feed_forward_dim,
            "dropout": self.dropout,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelConfig:
        fields = (
            "block_size",
            "decoder_blocks",
            "attention_heads",
            "embedding_dim",
            "feed_forward_dim",
            "dropout",
        )
        if set(value) != set(fields):
            raise ValueError("invalid model configuration fields")
        raw_integers = {name: value[name] for name in fields[:-1]}
        if not all(type(item) is int for item in raw_integers.values()):
            raise ValueError("invalid model configuration integer")
        dropout = value["dropout"]
        if type(dropout) is not float and type(dropout) is not int:
            raise ValueError("invalid model configuration dropout")
        config = cls(
            block_size=raw_integers["block_size"],  # type: ignore[arg-type]
            decoder_blocks=raw_integers["decoder_blocks"],  # type: ignore[arg-type]
            attention_heads=raw_integers["attention_heads"],  # type: ignore[arg-type]
            embedding_dim=raw_integers["embedding_dim"],  # type: ignore[arg-type]
            feed_forward_dim=raw_integers["feed_forward_dim"],  # type: ignore[arg-type]
            dropout=float(dropout),
        )
        config.validate()
        return config


@dataclass(frozen=True)
class EncodedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    target_byte_count: int


@dataclass(frozen=True)
class EvaluationMetrics:
    exact_accuracy: Accuracy
    bits_per_byte: float


@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    epochs: int = 20
    batch_size: int = 32
    seed: int = 42
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    device: DevicePreference = "auto"

    @classmethod
    def tiny(
        cls,
        *,
        epochs: int = 2,
        batch_size: int = 8,
        seed: int = 42,
        device: DevicePreference = "cpu",
    ) -> TrainingConfig:
        return cls(
            model=ModelConfig(
                block_size=64,
                decoder_blocks=1,
                attention_heads=2,
                embedding_dim=32,
                feed_forward_dim=64,
                dropout=0.1,
            ),
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )

    def validate(self) -> None:
        self.model.validate()
        if type(self.epochs) is not int or not 1 <= self.epochs <= 20:
            raise ValueError("epochs must be between 1 and 20")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")


@dataclass
class TrainingResult:
    model: ByteDecoderTransformer
    config: TrainingConfig
    device: torch.device
    best_epoch: int
    best_validation_loss: float


class ByteTokenizer:
    """Lossless UTF-8 byte encoding with command and padding tokens."""

    boundary_token = COMMAND_BOUNDARY_TOKEN
    padding_token = PADDING_TOKEN
    vocab_size = VOCAB_SIZE

    def encode_command(self, command: str) -> list[int]:
        return list(command.encode("utf-8"))

    def serialize_context(self, context: Sequence[str]) -> list[int]:
        if len(context) != COMMAND_CONTEXT_SIZE:
            raise ValueError("a command context must contain exactly three commands")
        tokens: list[int] = []
        for command in context:
            tokens.extend(self.encode_command(command))
            tokens.append(self.boundary_token)
        return tokens

    def encode_example(self, example: Example, block_size: int) -> EncodedExample:
        windows = self.encode_windows(example, block_size)
        if len(windows) != 1:
            raise ValueError("example requires multiple context windows")
        return windows[0]

    def encode_windows(
        self, example: Example, block_size: int
    ) -> list[EncodedExample]:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if len(example.context) != COMMAND_CONTEXT_SIZE:
            raise ValueError("a training example must contain exactly three commands")
        context_tokens = self.serialize_context(example.context)
        target_tokens = self.encode_command(example.target)
        serialized = context_tokens + target_tokens + [self.boundary_token]

        target_start = len(context_tokens)
        target_end = target_start + len(target_tokens)
        windows: list[EncodedExample] = []
        for chunk_start in range(target_start, target_end + 1, block_size):
            chunk_end = min(chunk_start + block_size - 1, target_end)
            window_start = max(0, chunk_end - block_size)
            window = serialized[window_start : chunk_end + 1]
            labels = tuple(
                token
                if chunk_start <= window_start + index + 1 <= chunk_end
                else IGNORE_INDEX
                for index, token in enumerate(window[1:])
            )
            target_byte_count = max(
                0,
                min(chunk_end, target_end - 1)
                - max(chunk_start, target_start)
                + 1,
            )
            windows.append(
                EncodedExample(tuple(window[:-1]), labels, target_byte_count)
            )
        return windows

    def decode_bytes(self, values: Sequence[int]) -> str | None:
        try:
            return bytes(values).decode("utf-8")
        except UnicodeDecodeError:
            return None


class CausalSelfAttention(nn.Module):
    causal_mask: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_heads = config.attention_heads
        self.head_dim = config.embedding_dim // config.attention_heads
        self.query_key_value = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size, dtype=torch.bool)
            ),
        )

    def forward(self, values: Tensor) -> Tensor:
        batch_size, sequence_length, embedding_dim = values.shape
        query, key, value = self.query_key_value(values).chunk(3, dim=-1)
        query = query.view(
            batch_size, sequence_length, self.attention_heads, self.head_dim
        )
        key = key.view(batch_size, sequence_length, self.attention_heads, self.head_dim)
        value = value.view(
            batch_size, sequence_length, self.attention_heads, self.head_dim
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, sequence_length, embedding_dim)
        return cast(Tensor, self.dropout(self.output(attended)))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        values = values + cast(Tensor, self.attention(self.attention_norm(values)))
        return values + cast(
            Tensor, self.feed_forward(self.feed_forward_norm(values))
        )


class ByteDecoderTransformer(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        config.validate()
        super().__init__()
        self.config = config
        self.tokenizer = ByteTokenizer()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embedding_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.decoder_blocks)]
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.language_model_head = nn.Linear(
            config.embedding_dim, VOCAB_SIZE, bias=False
        )
        self.language_model_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence_length = input_ids.shape
        if sequence_length == 0 or sequence_length > self.config.block_size:
            raise ValueError("input sequence exceeds the model context window")
        positions = torch.arange(sequence_length, device=input_ids.device)
        values = self.token_embedding(input_ids) + self.position_embedding(positions)
        values = self.embedding_dropout(values)
        for block in self.blocks:
            values = block(values)
        return cast(Tensor, self.language_model_head(self.final_norm(values)))

    def loss(self, input_ids: Tensor, labels: Tensor) -> Tensor:
        logits = self(input_ids)
        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        return F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    @torch.no_grad()
    def generate(
        self,
        context: Sequence[str],
        *,
        max_new_bytes: int = MAX_GENERATED_BYTES,
    ) -> str | None:
        if max_new_bytes < 0:
            raise ValueError("max_new_bytes must not be negative")
        was_training = self.training
        self.eval()
        try:
            tokens = self.tokenizer.serialize_context(context)
            generated: list[int] = []
            stopped_at_boundary = False
            for _ in range(min(max_new_bytes, MAX_GENERATED_BYTES)):
                window = tokens[-self.config.block_size :]
                input_ids = torch.tensor(
                    [window], dtype=torch.long, device=next(self.parameters()).device
                )
                logits = self(input_ids)[0, -1]
                logits[self.tokenizer.padding_token] = -torch.inf
                token = int(torch.argmax(logits).item())
                if token == self.tokenizer.boundary_token:
                    stopped_at_boundary = True
                    break
                if token >= BYTE_VOCAB_SIZE:
                    return None
                generated.append(token)
                tokens.append(token)
            if not stopped_at_boundary:
                return None
            return self.tokenizer.decode_bytes(generated)
        finally:
            if was_training:
                self.train()


def resolve_device(preference: DevicePreference) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.empty(1, device="cuda")
    except (RuntimeError, OSError):
        cuda_available = False
    if cuda_available:
        return torch.device("cuda")
    return torch.device("cpu")


def train_transformer(
    train_examples: Sequence[Example],
    validation_examples: Sequence[Example],
    config: TrainingConfig,
) -> TrainingResult:
    if not train_examples or not validation_examples:
        raise ValueError("training and validation examples must not be empty")
    config.validate()
    _seed_everything(config.seed)
    device = resolve_device(config.device)
    model = ByteDecoderTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    best_state: dict[str, Tensor] | None = None
    best_validation_loss = math.inf
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        indexes = torch.randperm(len(train_examples), generator=generator).tolist()
        for start in range(0, len(indexes), config.batch_size):
            batch_examples = [
                train_examples[index]
                for index in indexes[start : start + config.batch_size]
            ]
            input_ids, labels = _tensor_batch(
                batch_examples, model.tokenizer, config.model.block_size, device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(input_ids, labels)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        validation_loss = _mean_loss(
            model,
            validation_examples,
            batch_size=config.batch_size,
            device=device,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(model, config, device, best_epoch, best_validation_loss)


def evaluate_transformer(
    model: ByteDecoderTransformer,
    examples: Sequence[Example],
    *,
    batch_size: int = 32,
) -> EvaluationMetrics:
    if not examples:
        raise ValueError("evaluation examples must not be empty")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        correct = 0
        for example in examples:
            prediction = model.generate(example.context)
            if prediction == example.target:
                correct += 1
        total_byte_nll, target_byte_count = _byte_loss_totals(
            model, examples, batch_size=batch_size, device=device
        )
        if target_byte_count == 0:
            raise ValueError("evaluation has no target bytes")
        return EvaluationMetrics(
            exact_accuracy=Accuracy(correct, len(examples)),
            bits_per_byte=total_byte_nll / (target_byte_count * math.log(2)),
        )
    finally:
        if was_training:
            model.train()


def _tensor_batch(
    examples: Sequence[Example],
    tokenizer: ByteTokenizer,
    block_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    encoded = [
        window
        for example in examples
        for window in tokenizer.encode_windows(example, block_size)
    ]
    width = max(len(item.input_ids) for item in encoded)
    input_ids = torch.full(
        (len(encoded), width), tokenizer.padding_token, dtype=torch.long, device=device
    )
    labels = torch.full(
        (len(encoded), width), IGNORE_INDEX, dtype=torch.long, device=device
    )
    for row, item in enumerate(encoded):
        input_ids[row, : len(item.input_ids)] = torch.tensor(
            item.input_ids, dtype=torch.long, device=device
        )
        labels[row, : len(item.labels)] = torch.tensor(
            item.labels, dtype=torch.long, device=device
        )
    return input_ids, labels


@torch.no_grad()
def _mean_loss(
    model: ByteDecoderTransformer,
    examples: Sequence[Example],
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        input_ids, labels = _tensor_batch(
            batch, model.tokenizer, model.config.block_size, device
        )
        logits = model(input_ids)
        losses = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        valid = labels.reshape(-1) != IGNORE_INDEX
        total_loss += float(losses[valid].sum().item())
        total_tokens += int(valid.sum().item())
    if total_tokens == 0:
        raise ValueError("evaluation has no target tokens")
    return total_loss / total_tokens


@torch.no_grad()
def _byte_loss_totals(
    model: ByteDecoderTransformer,
    examples: Sequence[Example],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, int]:
    total_loss = 0.0
    total_bytes = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        encoded = [
            window
            for example in batch
            for window in model.tokenizer.encode_windows(
                example, model.config.block_size
            )
        ]
        input_ids, labels = _tensor_batch(
            batch, model.tokenizer, model.config.block_size, device
        )
        logits = model(input_ids)
        losses = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).reshape(labels.shape)
        byte_labels = (labels >= 0) & (labels < BYTE_VOCAB_SIZE)
        total_loss += float(losses[byte_labels].sum().item())
        total_bytes += sum(item.target_byte_count for item in encoded)
    return total_loss, total_bytes


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except (RuntimeError, AttributeError):
        pass
