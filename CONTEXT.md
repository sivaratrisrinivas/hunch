# Hunch

Hunch predicts the next shell command for one user from that user's recent command history.

## Language

**Command context**:
The three most recently executed commands supplied to Hunch when requesting a prediction.
_Avoid_: Prompt, input

**Suggestion**:
The exact text of one complete command predicted to follow the command context. A suggestion does nothing until the user copies it into the shell prompt and runs it.
_Avoid_: Completion, recommendation

**Acceptance**:
The user's choice to copy a suggestion into the shell prompt. Acceptance does not execute the command.
_Avoid_: Execution, selection

**Sensitive command**:
A command that appears to contain a password, token, private key, or other secret and must not enter Hunch's training data or suggestions.
_Avoid_: Private command

**First-run**:
The path from never having used Hunch to the first suggestion on screen. It has at most three steps. Arrival is seeing the suggestion, not accepting it.
_Avoid_: Onboarding, getting started

**Step**:
A user action that, if skipped, prevents the first suggestion from appearing. Waiting for training is a step. Getting a new prompt is a step when the hook needs one.
_Avoid_: Command, instruction

**Setup**:
The first-run action that trains on the user's history and installs the hook so the next prompt can show a suggestion.
_Avoid_: Init, bootstrap

**Daily use**:
A later terminal after first-run. A suggestion may appear with no further action.
_Avoid_: Session, recurring use

**Champion**:
The transformer weights Daily use uses to produce a suggestion, until an Update keeps new ones.
_Avoid_: Winner, production model, command-ngram

**Pile**:
The eight new usable commands that allow an Update to start.
_Avoid_: Batch, buffer, window

**Update**:
A later fit that continues from the Champion when a Pile exists. History is not rewritten.
_Avoid_: Fine-tune, retrain, online learning, realtime training

**Scoreboard**:
The command-context and next-command pairs held out at Setup. An Update is kept only if the transformer's exact-command accuracy on this set does not fall.
_Avoid_: Holdout, test set, validation set
