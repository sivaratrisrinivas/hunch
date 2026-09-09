# Hunch

Hunch learns command sequences from your Bash history. Given the three most
recent usable commands, it prints one likely next command. Hunch does not run
the command or send history over the network.

## What it does

The current version has two commands.

- `hunch train` reads Bash history, filters sensitive commands, evaluates two
  count-based predictors, and saves a command n-gram model.
- `hunch predict` uses the latest three usable history entries and prints at
  most one suggestion. It writes errors to standard error, so standard output
  contains only the suggestion or nothing.

Training treats each nonblank, non-timestamp physical history line as one
command. It preserves command order and splits the commands at the 80 percent
and 90 percent positions. The oldest portion trains the predictors. The next
portion is validation data, and the newest portion is test data.

Hunch reports exact-command accuracy for a most-common-command predictor and a
command n-gram predictor. The n-gram predictor first looks for the full
three-command context. If it has not seen that context, it tries two commands,
then one command, then the most common training command.

## Why this version uses counts

The count model tests the complete local workflow before Hunch adds a language
model. It establishes how Hunch reads and filters history, separates later
commands from training data, stores private state, handles failures, and prints
a suggestion for Bash to consume.

The two predictors provide comparison results for later models. Hunch will
evaluate the transformer with the same held-out exact-command accuracy metric.

## How to install and run it

Install Hunch in an isolated environment with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install .
hunch train
hunch predict
```

By default, Hunch reads `$HISTFILE`. If that variable is unset, it reads
`~/.bash_history`.

`hunch train` prints the number of usable and filtered commands, split sizes,
and validation and test accuracy. It replaces the saved model only after it has
read enough usable history and built a new model. A missing, empty, or short
history file leaves the previous model unchanged.

`hunch predict` reads history again, so its command context includes entries
saved after training. Inspect its output before you run it.

## How it handles private data

Hunch filters common password, token, private-key, authorization-header, and URL
credential patterns before it builds training examples. It does not write a
cleaned history file.

The filter cannot detect every secret. Configure Bash with
`HISTCONTROL=ignorespace` or `HISTCONTROL=ignoreboth`, then prefix a sensitive
command with a space to keep it out of future history. Hunch cannot read a
command that Bash did not save.

The saved count model contains command text. Hunch writes it to
`$XDG_DATA_HOME/hunch/count-model.json`, or
`~/.local/share/hunch/count-model.json` when `$XDG_DATA_HOME` is unset. The
directory uses mode `0700`, and the file uses mode `0600`. Hunch writes a
temporary file in the same private directory and atomically replaces the old
model after the write succeeds.

Tests and controlled environments can set `HUNCH_HISTORY_PATH` and
`HUNCH_STATE_DIR` to redirect both input and state.

## How to test it

```bash
uv sync
uv run pytest
uv run mypy
```
