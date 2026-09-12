# Hunch

Hunch reads the last three commands you ran and prints one guess for the next one.

It does not run that guess. It does not send your history anywhere.

## What

You already type the same command sequences. `git status`, then `git add .`, then `git commit`. After those, `git push` is a fair bet.

The guess comes from a small language model you train on your own Bash history, on one GPU on this machine. Same family as the big chat LLMs, much smaller job. It only reads your commands. It never calls ChatGPT or any other remote model. There is no API key.

After setup, a new prompt can print one guess. `Ctrl-X Ctrl-P` copies it onto an empty line. You still press Enter to run it. You still ignore it if it is wrong.

The trained model sits in `~/.local/share/hunch`. Closing the terminal does not delete it. Train again only when you want it to learn newer habits.

Lines that look like passwords, tokens, or keys never go into training and never come back as a guess. Prefix a secret command with a space so Bash never saves it. That is stronger than Hunch's filter, which is a precaution, not a lock.

## Why

If you have to go fetch a guess, you will type instead. Asking a hosted LLM at every prompt would be slower than that, and it would ship your history off the machine. A model trained on the internet also guesses generic Unix. Yours should guess you.

First-run is three steps. More than that, and people stop before they see a suggestion. Setup trains and installs the hook. You do not edit a file by hand. The next prompt is the product.

`Ctrl-R` only finds a line you already typed. Counting the most common command has the same limit. It can only replay something it has seen whole. A language model writes the next command one character at a time, so it can assemble a line that never appeared as one piece. That only helps if the guess shows up before you start typing. Hunch leaves a half-typed line alone.

Your history stays on the machine. Setup writes a Scoreboard of held-out command-context and next-command pairs into the state directory. That file is command text on purpose, so trimming `~/.bash_history` later does not change it. `hunch stats` still shows three integers: how many times you trained, how many guesses it showed, and how many you inserted.

On a laptop CPU, one guess can take more than 200 milliseconds. Waiting that long for every prompt is worse than no guess, so Hunch stops guessing and waits for `Ctrl-X Ctrl-P`. Open a new shell and it tries again. First-run itself does not use the CPU. If this machine has no GPU that PyTorch can use, setup stops and says so.

## How

You already have this repo. This machine already has one GPU that PyTorch can use.

```bash
uv tool install .
hunch setup
```

`hunch setup` trains on your history on that GPU, then writes the hook. Training prints nothing until it finishes. Then it tells you how often the model guessed the exact next command.

Open a new terminal, or `source ~/.bash_aliases` and press Enter. A guess may print above the prompt. If later prompts stay quiet, `Ctrl-X Ctrl-P` still asks for one.

```bash
hunch predict
hunch stats
```

Read the guess before you run it.
