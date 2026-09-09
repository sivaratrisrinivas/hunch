from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).parents[1]


def run_hunch(home: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        HOME=str(home),
        HISTFILE=str(home / ".bash_history"),
        HUNCH_HISTORY_PATH=str(home / ".bash_history"),
        HUNCH_STATE_DIR=str(home / ".local" / "share" / "hunch"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
    )
    return subprocess.run(
        [str(Path(sys.executable).with_name("hunch")), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def write_history(home: Path, commands: list[str]) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    history = home / ".bash_history"
    history.write_text("\n".join(commands) + "\n", encoding="utf-8")
    return history


def patterned_history(repetitions: int = 12) -> list[str]:
    commands: list[str] = []
    for _ in range(repetitions):
        commands.extend(["git status", "git add .", "git commit", "git push"])
    return commands


def test_train_and_predict_through_process_boundary(tmp_path: Path) -> None:
    home = tmp_path / "home"
    commands = patterned_history()
    write_history(home, ["#1700000000", "", *commands])

    trained = run_hunch(home, "train")

    assert trained.returncode == 0, trained.stderr
    assert "split: train=38 validation=5 test=5" in trained.stdout
    assert "validation most-common exact-command accuracy:" in trained.stdout
    assert "validation command-ngram exact-command accuracy:" in trained.stdout
    assert "test most-common exact-command accuracy:" in trained.stdout
    assert "test command-ngram exact-command accuracy:" in trained.stdout

    state_file = home / ".local" / "share" / "hunch" / "count-model.json"
    assert state_file.is_file()
    assert stat.S_IMODE(state_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(state_file.parent.stat().st_mode) == 0o700
    assert not any(
        path.is_file() and path != state_file
        for path in (home / ".local" / "share" / "hunch").iterdir()
    )

    write_history(home, [*commands, "git status", "git add .", "git commit"])
    predicted = run_hunch(home, "predict")
    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stdout == "git push\n"


def test_sensitive_commands_are_filtered_before_training(tmp_path: Path) -> None:
    home = tmp_path / "home"
    commands = patterned_history(15)
    commands[8:8] = [
        "export API_TOKEN=super-secret-value",
        "curl -H 'Authorization: Bearer abc123' https://example.test",
        "ssh -i ~/.ssh/id_rsa server",
        "https://admin:hunter2@example.test/private",
        "PGPASSWORD=hunter2 psql",
        "export AWS_SECRET_ACCESS_KEY=hunter2",
        "API-KEY=hunter2 deploy",
        "token:hunter2 command",
    ]
    write_history(home, commands)

    result = run_hunch(home, "train")

    assert result.returncode == 0, result.stderr
    model_text = (home / ".local/share/hunch/count-model.json").read_text()
    assert "super-secret-value" not in model_text
    assert "Authorization" not in model_text
    assert "id_rsa" not in model_text
    assert "hunter2" not in model_text
    assert "PGPASSWORD" not in model_text
    assert "AWS_SECRET_ACCESS_KEY" not in model_text
    assert "API-KEY" not in model_text
    assert "token:hunter2" not in model_text
    assert "filtered sensitive commands: 8" in result.stdout


def test_chronological_split_does_not_train_on_later_commands(tmp_path: Path) -> None:
    home = tmp_path / "home"
    early = ["old"] * 40
    late = ["future-validation"] * 5 + ["future-test"] * 5
    write_history(home, early + late)

    result = run_hunch(home, "train")

    assert result.returncode == 0, result.stderr
    model = json.loads(
        (home / ".local/share/hunch/count-model.json").read_text(encoding="utf-8")
    )
    serialized_counts = json.dumps(model["counts"])
    assert "future-validation" not in serialized_counts
    assert "future-test" not in serialized_counts
    assert model["most_common"] == "old"


def test_ngram_backoff_and_most_common_baselines_have_known_results(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    training_cycle = ["a", "b", "c", "d"] * 10
    held_out = ["x", "b", "c", "d"] * 2 + ["a", "b"]
    write_history(home, training_cycle + held_out)

    result = run_hunch(home, "train")

    assert result.returncode == 0, result.stderr
    assert "test most-common exact-command accuracy: 20.00% (1/5)" in result.stdout
    assert "test command-ngram exact-command accuracy: 80.00% (4/5)" in result.stdout


@pytest.mark.parametrize(
    ("history", "message"),
    [
        (None, "history file does not exist"),
        (["", "#1700000000", "   "], "history has no usable commands"),
        (["one", "two", "three", "four", "five"], "history is too small"),
    ],
)
def test_training_failure_preserves_existing_state(
    tmp_path: Path, history: list[str] | None, message: str
) -> None:
    home = tmp_path / "home"
    state = home / ".local/share/hunch/count-model.json"
    state.parent.mkdir(parents=True)
    state.write_text('{"existing": true}\n', encoding="utf-8")
    if history is not None:
        write_history(home, history)

    result = run_hunch(home, "train")

    assert result.returncode != 0
    assert message in result.stderr
    assert result.stdout == ""
    assert state.read_text(encoding="utf-8") == '{"existing": true}\n'


def test_predict_failure_is_clear_and_prints_no_suggestion(tmp_path: Path) -> None:
    home = tmp_path / "home"
    write_history(home, patterned_history())

    result = run_hunch(home, "predict")

    assert result.returncode != 0
    assert result.stdout == ""
    assert "model state does not exist" in result.stderr


def test_predict_rejects_a_control_character_suggestion(tmp_path: Path) -> None:
    home = tmp_path / "home"
    unsafe_suggestion = "echo unsafe\u0085text"
    cycle = ["one", "two", "three", unsafe_suggestion] * 12
    write_history(home, cycle)
    trained = run_hunch(home, "train")
    assert trained.returncode == 0, trained.stderr
    write_history(home, [*cycle, "one", "two", "three"])

    predicted = run_hunch(home, "predict")

    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stdout == ""
