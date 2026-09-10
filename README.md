# Hunch

Hunch learns command sequences from your Bash history. Given the three most
recent usable commands, it prints one likely next command. Hunch does not run
the command or send history over the network.

## What

Hunch now trains a byte-level decoder-only transformer and keeps the count
predictors as fixed baselines. `hunch train` reports both kinds of prediction
on validation and test data. `hunch predict` loads the transformer checkpoint
and prints one complete suggestion or nothing.

## Why

The count predictors are useful reference points, but they cannot generalize
within a command or to a command they have not seen. The transformer learns
the next UTF-8 byte from three earlier commands, so its metrics show whether
the extra model capacity helps on held-out history.

## How

Training filters sensitive history entries, splits the remaining commands in
time order, trains from random weights, and keeps the epoch with the lowest
validation loss. It then evaluates that checkpoint on the untouched test
portion and saves it atomically in the private state directory.

## Install and run

Install Hunch in an isolated environment with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install .
hunch train
hunch predict
```

By default, Hunch reads `$HISTFILE`. If that variable is unset, it reads
`~/.bash_history`.

Use `--device cpu` to force CPU training. The default `--device auto` uses CUDA
only when PyTorch reports a usable CUDA device. `--tiny` selects a small CPU
profile by default. `--epochs`, `--batch-size`, and `--seed` are also
available for controlled runs. Epochs are limited to 20.

`hunch predict` reads history again, so its command context includes entries
saved after training. It loads the best saved transformer checkpoint, generates
one suggestion with greedy byte-level decoding, and prints that command or
nothing. Generation stops at the end-of-command token or after 256 generated
bytes. A run that hits the byte limit without an end token is discarded as
incomplete. If the serialized command context is longer than the model window,
Hunch keeps the most recent bytes. Blank, multiline, oversized, invalid,
control-character, and sensitive suggestions are discarded.

Prediction runs on the CPU. It uses CUDA only when the checkpoint can be moved
onto a PyTorch CUDA device that is actually usable. Inspect the suggestion
before you run it.

## Training data

Hunch treats each nonblank, non-timestamp physical history line as one
command. It filters common password, token, private-key, authorization-header,
and URL-credential patterns before building examples. It preserves the
remaining command order and does not write a cleaned history file.

The chronological split uses the oldest 80 percent for training, the next 10
percent for validation, and the newest 10 percent for the untouched test. Each
target uses the three commands immediately before it. Context commands may
cross a split boundary because they were already present when the target ran.
Validation chooses the checkpoint. The test portion is evaluated after that
choice and does not affect training.

## Byte-level transformer

The transformer maps each UTF-8 byte directly to token IDs 0 through 255. It
uses token 256 as a command boundary and token 257 for padding. It needs no
unknown, beginning-of-sequence, or subword token.

For three context commands and a target command, the token stream is:

```text
bytes(command 1), boundary,
bytes(command 2), boundary,
bytes(command 3), boundary,
bytes(target), boundary
```

The model receives each token and predicts the next token. Loss labels for the
context are `-100`, which PyTorch ignores. The first target byte is therefore
predicted from the final context boundary. The target boundary is included in
the loss so the model learns when to stop. The model uses a 256-token causal
window. Long targets are split across windows so every target byte receives a
loss.

The default model has four decoder blocks, four attention heads, 128-wide
embeddings, 512-wide feed-forward layers, learned positions, tied input and
output embeddings, LayerNorm, residual connections, and 0.1 dropout. The
implementation uses ordinary PyTorch tensor operations for causal attention.

Training uses a fixed seed and AdamW for at most 20 epochs. It keeps the model
state with the lowest validation loss in memory. It evaluates exact-command
accuracy with greedy generation. It reports bits per byte from target-byte
cross-entropy. The terminating boundary is excluded from that metric's
numerator and denominator.

The most-common and command-ngram predictors remain fixed count baselines.
They are fitted only on the training commands and are reported beside the
transformer on validation and test data. They are not saved as production
state.

## Private state and failures

The transformer checkpoint is stored at
`$XDG_DATA_HOME/hunch/transformer.pt`, or
`~/.local/share/hunch/transformer.pt` when `$XDG_DATA_HOME` is unset. The
checkpoint contains the model configuration, tokenizer metadata, and CPU
weights in one file. The directory uses mode `0700`, and the checkpoint uses
mode `0600`.

Hunch writes a unique temporary checkpoint in the same private directory,
flushes and syncs it, validates it through the normal loader, and atomically
replaces the old checkpoint. Training does not publish until validation and
test evaluation succeed. A failed or interrupted run therefore leaves the
previous checkpoint unchanged.

Hunch loads checkpoints onto the CPU first and validates their version,
architecture, tokenizer metadata, configuration, and state dictionary. An
unreadable or legacy count-model state produces a clear error and never falls
back to random weights.

Configure Bash with `HISTCONTROL=ignorespace` or `HISTCONTROL=ignoreboth`, then
prefix a sensitive command with a space to keep it out of future history.
Hunch cannot read a command that Bash did not save.

Tests and controlled environments can set `HUNCH_HISTORY_PATH` and
`HUNCH_STATE_DIR` to redirect input and state.

## Development

```bash
uv sync
uv run pytest
uv run mypy
```
