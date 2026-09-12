from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from hunch.scoreboard import SCOREBOARD_FILENAME
from hunch.state import STATE_FILENAME
from hunch.stats import STATS_FILENAME
from tests.scripted import install_scripted_checkpoint
from tests.test_cli import patterned_history, write_history
from tests.test_shell import CONTEXT, SUGGESTION


PROJECT_ROOT = Path(__file__).parents[1]
PERSONAL_CANARY = "PERSONAL_HISTORY_CANARY_do_not_leak"
ALIASES_LINE = 'eval "$(hunch shell-init)"'
SKIP_REPO_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    "__pycache__",
}


@pytest.fixture(scope="module")
def installed_hunch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("uv-tool")
    tool_dir = root / "tools"
    bin_dir = root / "bin"
    venv_cfg = Path(sys.prefix) / "pyvenv.cfg"
    before = venv_cfg.read_bytes() if venv_cfg.is_file() else None
    installed = subprocess.run(
        ["uv", "tool", "install", "--force", "--refresh", str(PROJECT_ROOT)],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "UV_TOOL_DIR": str(tool_dir),
            "UV_TOOL_BIN_DIR": str(bin_dir),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert installed.returncode == 0, installed.stderr
    hunch = bin_dir / "hunch"
    assert hunch.is_file(), installed.stdout + installed.stderr
    if before is not None:
        assert venv_cfg.read_bytes() == before
    return hunch


def run_installed(
    hunch: Path,
    home: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(hunch), *arguments],
        cwd=home,
        env=_command_env(hunch, home),
        text=True,
        capture_output=True,
        check=False,
    )


def run_login_style_bash(
    hunch: Path, home: Path, script: str, *, path_prefix: Path | None = None
) -> subprocess.CompletedProcess[str]:
    environment = _command_env(hunch, home)
    if path_prefix is not None:
        environment["PATH"] = f"{path_prefix}:{environment['PATH']}"
    return subprocess.run(
        ["bash", "--norc", "--noprofile"],
        input=script,
        cwd=home,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _command_env(hunch: Path, home: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        HOME=str(home),
        HISTFILE=str(home / ".bash_history"),
        HUNCH_HISTORY_PATH=str(home / ".bash_history"),
        HUNCH_STATE_DIR=str(home / ".local" / "share" / "hunch"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        PATH=f"{hunch.parent}:{environment.get('PATH', '')}",
        CUDA_VISIBLE_DEVICES="",
    )
    return environment


def repo_state_files() -> set[Path]:
    found: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        dirnames[:] = [name for name in dirnames if name not in SKIP_REPO_DIRS]
        for filename in filenames:
            if filename in {SCOREBOARD_FILENAME, STATE_FILENAME, STATS_FILENAME}:
                found.add(Path(dirpath) / filename)
    return found


def test_uv_tool_install_uses_an_isolated_command(installed_hunch: Path) -> None:
    resolved = installed_hunch.resolve()
    venv_hunch = Path(sys.executable).with_name("hunch").resolve()
    python = resolved.parent / "python"
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import sys, hunch, torch; "
            "print(sys.prefix); print(hunch.__file__); "
            "print(torch.__version__); print(torch.version.cuda)",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert resolved != venv_hunch
    assert python.is_file()
    assert probe.returncode == 0, probe.stderr
    prefix, package, torch_version, cuda_version = probe.stdout.splitlines()
    assert Path(prefix).resolve() != Path(sys.prefix).resolve()
    assert str(PROJECT_ROOT / "src") not in package
    assert package.endswith("site-packages/hunch/__init__.py")
    assert torch_version.endswith("+cu121")
    assert cuda_version == "12.1"


def test_installed_command_trains_predicts_and_keeps_state_private(
    installed_hunch: Path, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    commands = patterned_history()
    write_history(
        home,
        [
            "#1700000000",
            "",
            f"export API_TOKEN={PERSONAL_CANARY}",
            *commands,
        ],
    )
    (home / ".not-history").write_text(f"{PERSONAL_CANARY}\n", encoding="utf-8")
    before_repo = repo_state_files()

    trained = run_installed(
        installed_hunch, home, "train", "--tiny", "--device", "cpu"
    )

    assert trained.returncode == 0, trained.stderr
    assert trained.stderr == ""
    assert "device: cpu" in trained.stdout
    assert "validation most-common exact-command accuracy:" in trained.stdout
    assert "validation command-ngram exact-command accuracy:" in trained.stdout
    assert "test transformer bits per byte:" in trained.stdout
    assert PERSONAL_CANARY not in trained.stdout
    assert PERSONAL_CANARY not in trained.stderr

    state_dir = home / ".local" / "share" / "hunch"
    checkpoint = state_dir / STATE_FILENAME
    scoreboard = state_dir / SCOREBOARD_FILENAME
    stats = state_dir / STATS_FILENAME
    assert checkpoint.is_file()
    assert scoreboard.is_file()
    assert stats.is_file()
    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600
    assert stat.S_IMODE(scoreboard.stat().st_mode) == 0o600
    assert stat.S_IMODE(stats.stat().st_mode) == 0o600
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
    stored_scoreboard = scoreboard.read_text(encoding="utf-8")
    assert "git status" in stored_scoreboard
    assert PERSONAL_CANARY not in stored_scoreboard
    assert repo_state_files() == before_repo
    assert checkpoint.resolve().is_relative_to(home.resolve())
    assert not checkpoint.resolve().is_relative_to(PROJECT_ROOT.resolve())

    write_history(home, [*commands, "git status", "git add .", "git commit"])
    predicted = run_installed(installed_hunch, home, "predict")
    counted = run_installed(installed_hunch, home, "stats")

    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stderr == ""
    assert predicted.stdout.count("\n") <= 1
    assert PERSONAL_CANARY not in predicted.stdout
    assert PERSONAL_CANARY not in predicted.stderr
    assert counted.returncode == 0, counted.stderr
    assert counted.stdout == (
        "training runs: 1\n"
        "suggestions displayed: 0\n"
        "suggestions inserted: 0\n"
    )
    stored = stats.read_text(encoding="utf-8")
    assert PERSONAL_CANARY not in stored
    assert "git status" not in stored
    assert "git push" not in stored


def test_aliases_startup_line_displays_inserts_and_falls_back(
    installed_hunch: Path, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    write_history(home, list(CONTEXT))
    install_scripted_checkpoint(home, CONTEXT, SUGGESTION.encode())
    (home / ".bash_aliases").write_text(f"{ALIASES_LINE}\n", encoding="utf-8")
    source_aliases = (
        '[ -f "$HOME/.bash_aliases" ] && . "$HOME/.bash_aliases"\n'
    )

    result = run_login_style_bash(
        installed_hunch,
        home,
        source_aliases
        + 'eval "$PROMPT_COMMAND"\n'
        "insert=$(bind -X | sed -n 's/.*\"\\([^\\\"]*\\)\"$/\\1/p')\n"
        "READLINE_LINE=\n"
        "READLINE_POINT=0\n"
        '"$insert"\n'
        'printf "LINE=%s\\n" "$READLINE_LINE"\n'
        "hunch stats\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == SUGGESTION
    assert f"LINE={SUGGESTION}" in result.stdout
    assert "training runs: 0" in result.stdout
    assert "suggestions displayed: 1" in result.stdout
    assert "suggestions inserted: 1" in result.stdout
    assert PERSONAL_CANARY not in result.stdout
    assert PERSONAL_CANARY not in result.stderr

    log = tmp_path / "hunch-calls.log"
    wrapper_dir = tmp_path / "wrapper"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "hunch"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$1\" >> {log.as_posix()!r}\n"
        "if [[ $1 == predict ]]; then sleep 0.25; fi\n"
        f'exec {installed_hunch.as_posix()!r} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)

    slow = run_login_style_bash(
        installed_hunch,
        home,
        source_aliases
        + 'eval "$PROMPT_COMMAND" >first.out\n'
        'eval "$PROMPT_COMMAND" >second.out\n'
        "insert=$(bind -X | sed -n 's/.*\"\\([^\\\"]*\\)\"$/\\1/p')\n"
        "READLINE_LINE=\n"
        "READLINE_POINT=0\n"
        '"$insert"\n'
        'printf "FIRST="\n'
        "cat first.out\n"
        'printf "SECOND="\n'
        "cat second.out\n"
        'printf "LINE=%s\\n" "$READLINE_LINE"\n',
        path_prefix=wrapper_dir,
    )

    assert slow.returncode == 0, slow.stderr
    assert slow.stdout == f"FIRST={SUGGESTION}\nSECOND=LINE={SUGGESTION}\n"
    assert log.read_text(encoding="utf-8").splitlines().count("predict") == 2
