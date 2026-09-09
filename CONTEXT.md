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
