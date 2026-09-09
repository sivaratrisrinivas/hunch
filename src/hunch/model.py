from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


MAX_ORDER = 3


@dataclass(frozen=True)
class Example:
    context: tuple[str, ...]
    target: str


@dataclass(frozen=True)
class Split:
    train_commands: list[str]
    validation: list[Example]
    test: list[Example]


@dataclass(frozen=True)
class Accuracy:
    correct: int
    total: int

    @property
    def percent(self) -> float:
        return 100 * self.correct / self.total


class CountModel:
    def __init__(
        self,
        most_common: str,
        counts: Mapping[int, Mapping[tuple[str, ...], Mapping[str, int]]],
    ) -> None:
        self.most_common = most_common
        self.counts = {
            order: {
                context: dict(target_counts)
                for context, target_counts in contexts.items()
            }
            for order, contexts in counts.items()
        }

    @classmethod
    def train(cls, commands: Sequence[str]) -> CountModel:
        target_counts = Counter(commands)
        most_common = _stable_winner(target_counts)
        counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            order: defaultdict(Counter) for order in range(1, MAX_ORDER + 1)
        }
        for target_index in range(1, len(commands)):
            for order in range(1, min(MAX_ORDER, target_index) + 1):
                context = tuple(commands[target_index - order : target_index])
                counts[order][context][commands[target_index]] += 1
        return cls(most_common, counts)

    def predict(self, context: Sequence[str]) -> str:
        for order in range(min(MAX_ORDER, len(context)), 0, -1):
            target_counts = self.counts.get(order, {}).get(tuple(context[-order:]))
            if target_counts:
                return _stable_winner(target_counts)
        return self.most_common


def chronological_split(commands: Sequence[str]) -> Split:
    if len(commands) < 10:
        raise ValueError(
            "history is too small: need at least 10 usable commands for an 80/10/10 split"
        )
    train_end = int(len(commands) * 0.8)
    validation_end = int(len(commands) * 0.9)
    return Split(
        train_commands=list(commands[:train_end]),
        validation=_examples(commands, train_end, validation_end),
        test=_examples(commands, validation_end, len(commands)),
    )


def evaluate(predictions: Iterable[str], examples: Sequence[Example]) -> Accuracy:
    correct = sum(
        prediction == example.target
        for prediction, example in zip(predictions, examples, strict=True)
    )
    return Accuracy(correct, len(examples))


def evaluate_constant(command: str, examples: Sequence[Example]) -> Accuracy:
    return evaluate((command for _ in examples), examples)


def evaluate_model(model: CountModel, examples: Sequence[Example]) -> Accuracy:
    return evaluate((model.predict(example.context) for example in examples), examples)


def _stable_winner(counts: Mapping[str, int]) -> str:
    first_seen = {command: index for index, command in enumerate(counts)}
    return max(counts, key=lambda command: (counts[command], -first_seen[command]))


def _examples(commands: Sequence[str], start: int, end: int) -> list[Example]:
    return [
        Example(tuple(commands[index - MAX_ORDER : index]), commands[index])
        for index in range(start, end)
    ]
