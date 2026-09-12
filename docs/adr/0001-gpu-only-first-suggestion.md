# GPU-only first suggestion

First-run trains on the user's Bash history on one local GPU. A hosted GPU would send history off the machine. A CPU fit is too slow to be a first-run step, and the count table is not what `predict` shows. If PyTorch cannot place tensors on that GPU, setup stops and says so.
