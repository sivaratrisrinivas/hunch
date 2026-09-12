from __future__ import annotations

import os

from hunch.cuda_env import prepare_local_cuda_libraries


def test_prepare_local_cuda_libraries_drops_windows_cuda_9(monkeypatch) -> None:
    monkeypatch.setenv(
        "PATH",
        "/usr/bin:/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/bin:/bin",
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/other")
    prepare_local_cuda_libraries()
    assert "CUDA/v9.0" not in os.environ["PATH"]
    assert "/usr/bin" in os.environ["PATH"]
    if os.path.isdir("/usr/lib/wsl/lib"):
        assert os.environ["LD_LIBRARY_PATH"].startswith("/usr/lib/wsl/lib")
