from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from tests.scripted import install_scripted_checkpoint
from tests.test_cli import patterned_history, run_hunch, write_history


HUNCH_BIN = Path(sys.executable).with_name("hunch")
CONTEXT = ("ls", "cd src", "git status")
SUGGESTION = "git push"
STATS_PATH = Path(".local/share/hunch/stats.json")


def run_bash(
    home: Path,
    script: str,
    *,
    path_prefix: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        HOME=str(home),
        HISTFILE=str(home / ".bash_history"),
        HUNCH_HISTORY_PATH=str(home / ".bash_history"),
        HUNCH_STATE_DIR=str(home / ".local" / "share" / "hunch"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
    )
    path = f"{HUNCH_BIN.parent}:{environment.get('PATH', '')}"
    if path_prefix is not None:
        path = f"{path_prefix}:{path}"
    environment["PATH"] = path
    return subprocess.run(
        ["bash", "--norc", "--noprofile"],
        input=script,
        cwd=home,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def source_hunch(body: str) -> str:
    return f"eval \"$(hunch shell-init)\"\n{body}\n"


def install_slow_hunch(directory: Path, log: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    wrapper = directory / "hunch"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$1\" >> {log.as_posix()!r}\n"
        'if [[ $1 == predict ]]; then sleep 0.25; fi\n'
        f'exec {HUNCH_BIN.as_posix()!r} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper


def prepared_home(tmp_path: Path, suggestion: str = SUGGESTION) -> Path:
    home = tmp_path / "home"
    write_history(home, list(CONTEXT))
    install_scripted_checkpoint(home, CONTEXT, suggestion.encode())
    return home


def test_shell_init_emits_inspectable_bash_without_modifying_the_shell(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    emitted = run_hunch(home, "shell-init")

    assert emitted.returncode == 0, emitted.stderr
    assert emitted.stderr == ""
    assert "PROMPT_COMMAND" in emitted.stdout
    assert r"\C-x\C-p" in emitted.stdout
    syntax = subprocess.run(
        ["bash", "--norc", "--noprofile", "-n"],
        input=emitted.stdout,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    probe = run_bash(
        home,
        "PROMPT_COMMAND=original\n"
        "hunch shell-init >/dev/null\n"
        'printf "%s\\n" "$PROMPT_COMMAND"\n',
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout == "original\n"


def test_integration_registers_hook_and_composes_with_prompt_command(
    tmp_path: Path,
) -> None:
    home = prepared_home(tmp_path)
    result = run_bash(
        home,
        "PROMPT_COMMAND='printf EXISTING\\n'\n"
        + source_hunch(
            'printf "PROMPT=%s\\n" "$PROMPT_COMMAND"\n'
            "bind -X\n"
            'eval "$PROMPT_COMMAND"\n'
            "unset _HUNCH_LOADED\n"
            "PROMPT_COMMAND=('printf ARRAY\\n')\n"
            'eval "$(hunch shell-init)"\n'
            'printf "ARRAY_COUNT=%s\\n" "${#PROMPT_COMMAND[@]}"\n'
            'for hook in "${PROMPT_COMMAND[@]}"; do eval "$hook"; done\n'
        ),
    )

    assert result.returncode == 0, result.stderr
    assert "EXISTING" in result.stdout
    assert "PROMPT=" in result.stdout
    assert "EXISTING" in result.stdout.split("PROMPT=", 1)[1].splitlines()[0]
    assert r'"\C-x\C-p"' in result.stdout
    assert SUGGESTION in result.stdout
    assert "ARRAY_COUNT=2" in result.stdout
    assert "ARRAY" in result.stdout


def test_successful_prediction_appears_on_its_own_line_above_the_prompt(
    tmp_path: Path,
) -> None:
    home = prepared_home(tmp_path)
    result = run_bash(
        home,
        source_hunch('eval "$PROMPT_COMMAND"\nprintf "PROMPT\\n"\n'),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{SUGGESTION}\nPROMPT\n"


def test_ctrl_x_ctrl_p_inserts_saved_suggestion_only_when_line_is_empty(
    tmp_path: Path,
) -> None:
    home = prepared_home(tmp_path)
    result = run_bash(
        home,
        source_hunch(
            'eval "$PROMPT_COMMAND" >/dev/null\n'
            "insert=$(bind -X | sed -n 's/.*\"\\([^\\\"]*\\)\"$/\\1/p')\n"
            "READLINE_LINE=\n"
            "READLINE_POINT=0\n"
            '"$insert"\n'
            'printf "EMPTY=%s\\n" "$READLINE_LINE"\n'
            'printf "POINT=%s\\n" "$READLINE_POINT"\n'
            "READLINE_LINE='keep this'\n"
            "READLINE_POINT=4\n"
            '"$insert"\n'
            'printf "NONEMPTY=%s\\n" "$READLINE_LINE"\n'
            'printf "NONEMPTY_POINT=%s\\n" "$READLINE_POINT"\n'
        ),
    )

    assert result.returncode == 0, result.stderr
    assert f"EMPTY={SUGGESTION}" in result.stdout
    assert f"POINT={len(SUGGESTION)}" in result.stdout
    assert "NONEMPTY=keep this" in result.stdout
    assert "NONEMPTY_POINT=4" in result.stdout


def test_accepting_a_suggestion_does_not_run_it(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    home = prepared_home(tmp_path, f"touch {marker}")
    result = run_bash(
        home,
        source_hunch(
            'eval "$PROMPT_COMMAND" >/dev/null\n'
            "insert=$(bind -X | sed -n 's/.*\"\\([^\\\"]*\\)\"$/\\1/p')\n"
            "READLINE_LINE=\n"
            "READLINE_POINT=0\n"
            '"$insert"\n'
            'printf "LINE=%s\\n" "$READLINE_LINE"\n'
        ),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"LINE=touch {marker}\n"
    assert not marker.exists()


def test_prediction_errors_and_empty_results_stay_silent(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    write_history(missing, list(CONTEXT))
    missing_result = run_bash(
        missing,
        source_hunch(
            "prompt_err=$(mktemp)\n"
            'eval "$PROMPT_COMMAND" >prompt.out 2>"$prompt_err"\n'
            "status=$?\n"
            "printf 'MISSING_OUT='\n"
            "cat prompt.out\n"
            "printf '\\nMISSING_ERR='\n"
            "cat \"$prompt_err\"\n"
            'printf "\\nMISSING_STATUS=%s\\n" "$status"\n'
        ),
    )

    empty = prepared_home(tmp_path, "")
    empty_result = run_bash(
        empty,
        source_hunch(
            "prompt_err=$(mktemp)\n"
            'eval "$PROMPT_COMMAND" >prompt.out 2>"$prompt_err"\n'
            "printf 'EMPTY_OUT='\n"
            "cat prompt.out\n"
            "printf '\\nEMPTY_ERR='\n"
            "cat \"$prompt_err\"\n"
            "printf '\\n'\n"
        ),
    )

    assert missing_result.returncode == 0, missing_result.stderr
    assert "MISSING_OUT=\n" in missing_result.stdout
    assert "MISSING_ERR=\n" in missing_result.stdout
    assert "MISSING_STATUS=0" in missing_result.stdout
    assert empty_result.returncode == 0, empty_result.stderr
    assert "EMPTY_OUT=\n" in empty_result.stdout
    assert "EMPTY_ERR=\n" in empty_result.stdout


def test_slow_prediction_disables_automatic_and_keeps_on_demand(
    tmp_path: Path,
) -> None:
    home = prepared_home(tmp_path)
    log = tmp_path / "hunch-calls.log"
    wrapper_dir = tmp_path / "wrapper"
    install_slow_hunch(wrapper_dir, log)
    result = run_bash(
        home,
        source_hunch(
            'eval "$PROMPT_COMMAND" >first.out\n'
            'eval "$PROMPT_COMMAND" >second.out\n'
            "insert=$(bind -X | sed -n 's/.*\"\\([^\\\"]*\\)\"$/\\1/p')\n"
            "READLINE_LINE=\n"
            "READLINE_POINT=0\n"
            '"$insert"\n'
            'printf "FIRST="\n'
            "cat first.out\n"
            'printf "SECOND="\n'
            "cat second.out\n"
            'printf "LINE=%s\\n" "$READLINE_LINE"\n'
        ),
        path_prefix=wrapper_dir,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"FIRST={SUGGESTION}\nSECOND=LINE={SUGGESTION}\n"
    commands = log.read_text(encoding="utf-8").splitlines()
    assert commands.count("predict") == 2


def test_stats_report_only_aggregate_counts_and_store_no_command_text(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    write_history(home, patterned_history())
    before = run_hunch(home, "stats")
    trained = run_hunch(home, "train", "--tiny", "--epochs", "1", "--device", "cpu")
    after_train = run_hunch(home, "stats")

    display_home = prepared_home(tmp_path / "display")
    used = run_bash(
        display_home,
        source_hunch(
            'eval "$PROMPT_COMMAND" >/dev/null\n'
            "insert=$(bind -X | sed -n 's/.*\"\\([^\\\"]*\\)\"$/\\1/p')\n"
            "READLINE_LINE=\n"
            '"$insert"\n'
        ),
    )
    displayed = run_hunch(display_home, "stats")

    assert before.returncode == 0, before.stderr
    assert before.stdout == (
        "training runs: 0\n"
        "suggestions displayed: 0\n"
        "suggestions inserted: 0\n"
    )
    assert trained.returncode == 0, trained.stderr
    assert after_train.returncode == 0, after_train.stderr
    assert after_train.stdout == (
        "training runs: 1\n"
        "suggestions displayed: 0\n"
        "suggestions inserted: 0\n"
    )
    assert used.returncode == 0, used.stderr
    assert displayed.returncode == 0, displayed.stderr
    assert displayed.stdout == (
        "training runs: 0\n"
        "suggestions displayed: 1\n"
        "suggestions inserted: 1\n"
    )

    payload = json.loads((display_home / STATS_PATH).read_text(encoding="utf-8"))
    assert payload == {
        "training_runs": 0,
        "suggestions_displayed": 1,
        "suggestions_inserted": 1,
    }
    stored = (display_home / STATS_PATH).read_text(encoding="utf-8")
    assert CONTEXT[0] not in stored
    assert CONTEXT[1] not in stored
    assert CONTEXT[2] not in stored
    assert SUGGESTION not in stored
    assert stat.S_IMODE((display_home / STATS_PATH).stat().st_mode) == 0o600
