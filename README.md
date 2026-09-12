# Hunch

Hunch reads the last three commands you ran and prints one guess for the next one.

It does not run that guess. It does not send your history anywhere.

## What

You already type the same command sequences. `git status`, then `git add .`, then `git commit`. After those, `git push` is a fair bet.

The guess comes from a small language model you train on your own Bash history. Same family as the big chat LLMs, much smaller job. It only reads your commands. It never calls ChatGPT or any other remote model. There is no API key.

After training, a new prompt can print one guess. `Ctrl-X Ctrl-P` copies it onto an empty line. You still press Enter to run it. You still ignore it if it is wrong.

The trained model sits in `~/.local/share/hunch`. Closing the terminal does not delete it. Train again only when you want it to learn newer habits.

Lines that look like passwords, tokens, or keys never go into training and never come back as a guess. Prefix a secret command with a space so Bash never saves it. That is stronger than Hunch's filter, which is a precaution, not a lock.

## Why

If you have to go fetch a guess, you will type instead. Asking a hosted LLM at every prompt would be slower than that, and it would ship your history off the machine. A model trained on the internet also guesses generic Unix. Yours should guess you.

`Ctrl-R` only finds a line you already typed. Counting the most common command has the same limit. It can only replay something it has seen whole. A language model writes the next command one character at a time, so it can assemble a line that never appeared as one piece. That only helps if the guess shows up before you start typing. Hunch leaves a half-typed line alone.

Your history stays on the machine. Hunch's counters are three integers. How many times you trained. How many guesses it showed. How many you inserted. Not the text.

On a laptop CPU, one guess can take more than 200 milliseconds. Waiting that long for every prompt is worse than no guess, so Hunch stops guessing and waits for `Ctrl-X Ctrl-P`. Open a new shell and it tries again.

## How

Install it, train the language model once, then tell Bash to load it.

```bash
uv tool install .
hunch train --device cpu
```

`hunch train` reads your history and shows the model three commands plus the one that came next. It keeps the version that does best on commands you ran later. Training prints nothing until it finishes. Then it tells you how often the model guessed the exact next command. On a CPU, a few thousand commands can take hours. That is normal.

Put this in `~/.bash_aliases`:

```bash
eval "$("$HOME/.local/bin/hunch" shell-init)"
```

Open a new terminal. Each prompt can ask that same local model for one next command. A guess may print above the prompt. If the next prompts stay quiet, `Ctrl-X Ctrl-P` still asks for one.

```bash
hunch predict
hunch stats
```

Read the guess before you run it.
