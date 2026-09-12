from __future__ import annotations

from argparse import Namespace
import os
from pathlib import Path
import stat
import subprocess
import sys

from pytest import MonkeyPatch

from hunch.cli import parser, setup
from hunch.scoreboard import SCOREBOARD_FILENAME
from hunch.shell import HOOK_ALIASES_LINE, install_hook
from tests.test_cli import (
    PROJECT_ROOT,
    holdout_pairs,
    patterned_history,
    scoreboard_pairs,
    write_history,
)


def test_install_hook_writes_the_aliases_line_once(tmp_path: Path) -> None:
    aliases = tmp_path / ".bash_aliases"
    aliases.write_text("alias ll='ls -alF'\n", encoding="utf-8")

    install_hook(aliases)
    install_hook(aliases)

    text = aliases.read_text(encoding="utf-8")
    assert HOOK_ALIASES_LINE in text
    assert text.count(HOOK_ALIASES_LINE) == 1
    assert text.startswith("alias ll='ls -alF'\n")


def test_setup_fails_when_no_local_gpu_is_usable(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    environment = os.environ.copy()
    environment.update(
        HOME=str(home),
        PYTHONPATH=str(PROJECT_ROOT / "src"),
        CUDA_VISIBLE_DEVICES="",
    )
    result = subprocess.run(
        [sys.executable, "-m", "hunch", "setup", "--tiny"],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "one local GPU" in result.stderr
    assert not (home / ".bash_aliases").exists()
    assert not (home / ".local" / "share" / "hunch").exists()


def test_setup_trains_then_writes_the_hook(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    trained: list[str] = []

    def fake_require() -> None:
        return None

    def fake_train(options: Namespace) -> int:
        trained.append(options.device)
        return 0

    monkeypatch.setattr("hunch.cli.require_cuda_device", fake_require)
    monkeypatch.setattr("hunch.cli.train", fake_train)

    assert setup(Namespace(tiny=True, epochs=None, batch_size=None, seed=None)) == 0
    assert trained == ["cuda"]
    assert HOOK_ALIASES_LINE in (home / ".bash_aliases").read_text(encoding="utf-8")


def test_setup_is_a_cli_command() -> None:
    options = parser().parse_args(["setup"])
    assert options.command == "setup"


def test_setup_writes_the_holdout_scoreboard_as_command_text(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    home = tmp_path / "home"
    commands = patterned_history()
    write_history(home, commands)
    state_directory = home / ".local" / "share" / "hunch"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HISTFILE", str(home / ".bash_history"))
    monkeypatch.setenv("HUNCH_HISTORY_PATH", str(home / ".bash_history"))
    monkeypatch.setenv("HUNCH_STATE_DIR", str(state_directory))
    monkeypatch.setattr("hunch.cli.require_cuda_device", lambda: None)

    assert setup(Namespace(tiny=True, epochs=1, batch_size=None, seed=None)) == 0

    scoreboard = state_directory / SCOREBOARD_FILENAME
    stored = scoreboard.read_text(encoding="utf-8")
    expected = holdout_pairs(commands)
    assert scoreboard_pairs(scoreboard) == expected
    assert "git status" in stored
    assert stat.S_IMODE(scoreboard.stat().st_mode) == 0o600
    assert HOOK_ALIASES_LINE in (home / ".bash_aliases").read_text(encoding="utf-8")

    write_history(home, commands[:8])
    assert scoreboard_pairs(scoreboard) == expected
