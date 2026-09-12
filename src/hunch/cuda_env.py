from __future__ import annotations

import os
from pathlib import Path


_WINDOWS_CUDA_9 = "/NVIDIA GPU Computing Toolkit/CUDA/v9.0"
_WSL_CUDA_LIB = Path("/usr/lib/wsl/lib")


def prepare_local_cuda_libraries() -> None:
    """Drop Windows CUDA 9.0 from PATH and prefer the WSL GPU driver libraries."""
    os.environ["PATH"] = ":".join(
        part
        for part in os.environ.get("PATH", "").split(":")
        if _WINDOWS_CUDA_9 not in part
    )
    if not _WSL_CUDA_LIB.is_dir():
        return
    current = [
        part
        for part in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if part and part != str(_WSL_CUDA_LIB)
    ]
    os.environ["LD_LIBRARY_PATH"] = ":".join([str(_WSL_CUDA_LIB), *current])
