from __future__ import annotations

import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest
import torch

from hunch.model import chronological_split
from hunch.pile import CONSUMED_FILENAME
from hunch.scoreboard import SCOREBOARD_FILENAME
from hunch.state import STATE_FILENAME
from hunch.stats import STATS_FILENAME
from hunch.transformer import (
    ARCHITECTURE_VERSION,
    TOKENIZER_VERSION,
)
from tests.scripted import install_scripted_checkpoint


PROJECT_ROOT = Path(__file__).parents[1]
HISTORY_MARKER = "UNIQUE_HISTORY_MARKER_do_not_leak"


def run_hunch(
    home: Path,
    *arguments: str,
    cuda_visible_devices: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        HOME=str(home),
        HISTFILE=str(home / ".bash_history"),
        HUNCH_HISTORY_PATH=str(home / ".bash_history"),
        HUNCH_STATE_DIR=str(home / ".local" / "share" / "hunch"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
    )
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
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

    trained = run_hunch(
        home,
        "train",
        "--tiny",
        "--device",
        "cuda",
        cuda_visible_devices="",
    )

    assert trained.returncode == 0, trained.stderr
    assert trained.stderr == ""
    assert "split: train=38 validation=5 test=5" in trained.stdout
    assert "device: cpu" in trained.stdout
    assert "validation most-common exact-command accuracy:" in trained.stdout
    assert "validation command-ngram exact-command accuracy:" in trained.stdout
    assert "test most-common exact-command accuracy:" in trained.stdout
    assert "test command-ngram exact-command accuracy:" in trained.stdout
    assert "test transformer bits per byte:" in trained.stdout

    state_file = home / ".local" / "share" / "hunch" / "transformer.pt"
    assert state_file.is_file()
    assert stat.S_IMODE(state_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(state_file.parent.stat().st_mode) == 0o700
    artifacts = {
        path.name
        for path in (home / ".local" / "share" / "hunch").iterdir()
        if path.is_file()
    }
    assert artifacts == {
        CONSUMED_FILENAME,
        SCOREBOARD_FILENAME,
        STATE_FILENAME,
        STATS_FILENAME,
    }
    scoreboard = home / ".local" / "share" / "hunch" / SCOREBOARD_FILENAME
    stored_scoreboard = scoreboard.read_bytes()
    assert stat.S_IMODE(scoreboard.stat().st_mode) == 0o600
    assert scoreboard_pairs(scoreboard) == holdout_pairs(commands)
    assert b"git status" in stored_scoreboard
    assert b"git commit" in stored_scoreboard

    write_history(home, commands[:8])
    assert scoreboard.read_bytes() == stored_scoreboard
    counted = run_hunch(home, "stats")
    assert counted.returncode == 0, counted.stderr
    assert counted.stdout == (
        "training runs: 1\n"
        "suggestions displayed: 0\n"
        "suggestions inserted: 0\n"
    )

    write_history(home, [*commands, "git status", "git add .", "git commit"])
    predicted = run_hunch(home, "predict")
    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stderr == ""
    assert predicted.stdout.count("\n") <= 1
    assert scoreboard.read_bytes() == stored_scoreboard


def test_sensitive_commands_are_filtered_before_training(tmp_path: Path) -> None:
    sensitive_commands = [
        "export API_TOKEN=super-secret-value",
        "curl -H 'Authorization: Bearer abc123' https://example.test",
        "ssh -i ~/.ssh/id_rsa server",
        "https://admin:hunter2@example.test/private",
        "PGPASSWORD=hunter2 psql",
        "export AWS_SECRET_ACCESS_KEY=hunter2",
        "API-KEY=hunter2 deploy",
        "token:hunter2 command",
    ]
    alternate_sensitive_commands = [
        "export API_TOKEN=another-secret-value",
        "curl -H 'Authorization: Bearer xyz789' https://example.test",
        "ssh -i ~/.ssh/id_rsa server",
        "https://admin:different-password@example.test/private",
        "PGPASSWORD=different-password psql",
        "export AWS_SECRET_ACCESS_KEY=different-secret",
        "API-KEY=different-key deploy",
        "token:different-token command",
    ]
    histories: list[list[str]] = []
    for replacement in (sensitive_commands, alternate_sensitive_commands):
        commands = patterned_history(15)
        commands[8:8] = replacement
        histories.append(commands)

    checkpoints: list[bytes] = []
    results: list[subprocess.CompletedProcess[str]] = []
    for index, history in enumerate(histories):
        home = tmp_path / f"home-{index}"
        write_history(home, history)
        result = run_hunch(home, "train", "--tiny", "--device", "cpu")
        results.append(result)
        checkpoints.append((home / ".local/share/hunch/transformer.pt").read_bytes())

    assert all(result.returncode == 0 for result in results)
    assert all(result.stderr == "" for result in results)
    assert all("filtered sensitive commands: 8" in result.stdout for result in results)
    assert checkpoints[0] == checkpoints[1]


def test_chronological_split_does_not_train_on_later_commands(tmp_path: Path) -> None:
    prefix = [f"early-{index}" for index in range(8)]
    histories = [
        prefix + ["validation-a", "test-a"],
        prefix + ["validation-b", "test-b"],
    ]
    checkpoints: list[bytes] = []
    for index, history in enumerate(histories):
        home = tmp_path / f"home-{index}"
        write_history(home, history)
        result = run_hunch(
            home, "train", "--tiny", "--epochs", "1", "--device", "cpu"
        )

        assert result.returncode == 0, result.stderr
        checkpoints.append((home / ".local/share/hunch/transformer.pt").read_bytes())

    assert checkpoints[0] == checkpoints[1]


def test_ngram_backoff_and_most_common_baselines_have_known_results(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    training_cycle = ["a", "b", "c", "d"] * 10
    held_out = ["x", "b", "c", "d"] * 2 + ["a", "b"]
    write_history(home, training_cycle + held_out)

    result = run_hunch(home, "train", "--tiny", "--device", "cpu")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
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


def test_failed_retraining_preserves_a_valid_checkpoint(tmp_path: Path) -> None:
    home = tmp_path / "home"
    write_history(home, patterned_history())

    trained = run_hunch(home, "train", "--tiny", "--device", "cpu")

    assert trained.returncode == 0, trained.stderr
    state = home / ".local/share/hunch/transformer.pt"
    original = state.read_bytes()

    write_history(home, ["one", "two", "three"])
    failed = run_hunch(home, "train", "--tiny", "--device", "cpu")

    assert failed.returncode != 0
    assert "history is too small" in failed.stderr
    assert state.read_bytes() == original


def test_predict_prints_one_undecorated_suggestion(tmp_path: Path) -> None:
    home = tmp_path / "home"
    context = ("ls", "cd src", "git status")
    write_history(home, [HISTORY_MARKER, "export API_TOKEN=secret", *context])
    install_scripted_checkpoint(home, context, b"git push")

    predicted = run_hunch(home, "predict", cuda_visible_devices="")

    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stderr == ""
    assert predicted.stdout == "git push\n"


def test_predict_keeps_the_most_recent_context_bytes(tmp_path: Path) -> None:
    home = tmp_path / "home"
    context = ("A" * 80, "B" * 80, "C" * 80)
    write_history(home, list(context))
    install_scripted_checkpoint(home, context, b"recent")

    predicted = run_hunch(home, "predict")

    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stderr == ""
    assert predicted.stdout == "recent\n"


@pytest.mark.parametrize(
    ("continuation", "emit_boundary"),
    [
        (b"", True),
        (b"   ", True),
        (b"echo a\necho b", True),
        (b"echo a\recho b", True),
        (b"A" * 8, False),
        (bytes([0xC0]), True),
        (b"echo \x07bell", True),
        (b"echo unsafe\xc2\x85text", True),
        (b"API_KEY=xyz", True),
    ],
    ids=[
        "blank",
        "whitespace",
        "newline",
        "carriage-return",
        "oversized",
        "invalid-utf8",
        "bell",
        "c1-control",
        "sensitive",
    ],
)
def test_predict_discards_unusable_suggestions(
    tmp_path: Path, continuation: bytes, emit_boundary: bool
) -> None:
    home = tmp_path / "home"
    context = ("one", "two", "three")
    write_history(home, list(context))
    install_scripted_checkpoint(
        home, context, continuation, emit_boundary=emit_boundary
    )

    predicted = run_hunch(home, "predict")

    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stderr == ""
    assert predicted.stdout == ""


def test_predict_failure_is_clear_and_prints_no_suggestion(tmp_path: Path) -> None:
    home = tmp_path / "home"
    write_history(home, [HISTORY_MARKER, *patterned_history()])

    result = run_hunch(home, "predict")

    assert result.returncode != 0
    assert result.stdout == ""
    assert "model state does not exist" in result.stderr
    assert HISTORY_MARKER not in result.stderr
    assert HISTORY_MARKER not in result.stdout


def test_predict_rejects_incompatible_and_unreadable_checkpoints(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    write_history(home, [HISTORY_MARKER, "ls", "cd src", "git status"])
    state_directory = home / ".local" / "share" / "hunch"
    state_directory.mkdir(parents=True)
    checkpoint = state_directory / STATE_FILENAME
    original = _incompatible_checkpoint_bytes()
    checkpoint.write_bytes(original)

    incompatible = run_hunch(home, "predict")

    assert incompatible.returncode != 0
    assert incompatible.stdout == ""
    assert "model state is unreadable" in incompatible.stderr
    assert HISTORY_MARKER not in incompatible.stderr
    assert checkpoint.read_bytes() == original

    garbage = b"<<<not-a-checkpoint>>>"
    checkpoint.write_bytes(garbage)
    unreadable = run_hunch(home, "predict")

    assert unreadable.returncode != 0
    assert unreadable.stdout == ""
    assert "model state is unreadable" in unreadable.stderr
    assert HISTORY_MARKER not in unreadable.stderr
    assert checkpoint.read_bytes() == garbage


def holdout_pairs(commands: list[str]) -> list[dict[str, object]]:
    return [
        {"context": list(example.context), "next": example.target}
        for example in chronological_split(commands).validation
    ]


def scoreboard_pairs(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    return payload


def _incompatible_checkpoint_bytes() -> bytes:
    payload = {
        "format_version": 999,
        "architecture": ARCHITECTURE_VERSION,
        "tokenizer": {
            "version": TOKENIZER_VERSION,
            "byte_ids": "identity-0-255",
            "command_boundary_id": 256,
            "padding_id": 257,
        },
        "model_config": {
            "block_size": 64,
            "decoder_blocks": 1,
            "attention_heads": 2,
            "embedding_dim": 32,
            "feed_forward_dim": 64,
            "dropout": 0.0,
        },
        "state_dict": {},
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()
