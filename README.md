# Hunch

Hunch predicts one likely next Bash command from the three most recent commands
in your local history. This first slice uses a command n-gram count model and a
most-common-command baseline. It never executes a suggestion or sends history
off the machine.

## Install and run

Install Hunch in an isolated environment with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install .
hunch train
hunch predict
```

`hunch train` reads `$HISTFILE`, falling back to `~/.bash_history`. It ignores
blank lines and Bash timestamp markers, filters common secret-looking commands,
keeps the remaining commands chronological, and reports both baselines on the
validation and test portions. It does not save a cleaned copy of history.

The model is stored under `$XDG_DATA_HOME/hunch` (normally
`~/.local/share/hunch`) with private permissions. Tests and controlled
environments can override the inputs with `HUNCH_HISTORY_PATH` and
`HUNCH_STATE_DIR`.

`hunch predict` prints either one exact command or nothing; diagnostics go to
standard error. Inspect the suggestion before running it.

## History privacy

The built-in sensitive-command filter is a precaution, not a guarantee. Bash
itself can keep especially sensitive commands out of history: with
`HISTCONTROL=ignorespace` or `HISTCONTROL=ignoreboth`, prefix a command with a
space. Hunch cannot learn commands Bash did not save.

The count-model state contains command text and must be treated as private.

## Development

```bash
uv sync
uv run pytest
uv run mypy
```
