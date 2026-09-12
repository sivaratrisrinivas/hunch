# Update continues from the Champion

After Setup, new usable commands form a Pile of eight. An Update continues from the last transformer checkpoint, mixes old commands with the new ones, and keeps the new weights only if exact-command accuracy on the Scoreboard does not fall. History is not rewritten.

The Scoreboard is the validation split taken at Setup and never re-split. Keep-or-discard uses the transformer's exact-command accuracy on that set only. Those pairs are stored as command text in the state directory so `HISTFILESIZE` cannot move the stick. `hunch stats` plus an update log in that directory show keep versus discard. Command-ngram still takes the Pile and still prints its number. It cannot become the Champion.

The prompt hook starts `hunch update` in the background when it sees a Pile. That is not a first-run step. If the GPU is missing or busy, command-ngram still takes the Pile. The transformer waits.

Retraining from scratch on every Pile was rejected as too slow. A single gradient step on only the new commands was rejected because the model forgets. Autoresearch's loop of rewriting the train file was rejected. This project is for watching one model learn, not for an agent to search architectures.
