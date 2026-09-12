# GPU-only first suggestion

First-run still trains on the user's Bash history on one local GPU. A hosted GPU would send history off the machine. A CPU fit is too slow to be a first-run step. If PyTorch cannot place tensors on that GPU, Setup stops and says so.

Daily use shows the transformer Champion. Command-ngram is scored and logged. It is never what `predict` shows. Setup still needs the GPU so the first Champion exists.
