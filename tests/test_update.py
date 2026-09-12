from __future__ import annotations

import json
from pathlib import Path

from hunch.model import Example
from hunch.scoreboard import SCOREBOARD_FILENAME, save_scoreboard
from hunch.state import STATE_FILENAME
from hunch.stats import STATS_FILENAME
from tests.scripted import install_scripted_checkpoint
from tests.test_cli import patterned_history, run_hunch, write_history


CONSUMED_FILENAME = "consumed.json"
UPDATE_LOG_FILENAME = "update.log"
CONTEXT = ("alpha", "beta", "gamma")
CHAMPION_SUGGESTION = "delta"


def state_dir(home: Path) -> Path:
    return home / ".local" / "share" / "hunch"


def write_consumed(home: Path, consumed: int) -> None:
    directory = state_dir(home)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CONSUMED_FILENAME).write_text(
        json.dumps({"consumed": consumed}) + "\n", encoding="utf-8"
    )


def install_champion(home: Path, old: list[str], pile: list[str]) -> Path:
    write_history(home, old + pile)
    write_consumed(home, len(old))
    return install_scripted_checkpoint(home, CONTEXT, CHAMPION_SUGGESTION.encode())


def test_update_before_setup_exits_and_writes_no_champion(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    result = run_hunch(home, "update")

    assert result.returncode == 2
    assert "no champion" in result.stderr.lower()
    assert "setup" in result.stderr.lower()
    assert result.stdout == ""
    assert not (state_dir(home) / STATE_FILENAME).exists()


def test_update_without_a_pile_does_not_change_the_champion(tmp_path: Path) -> None:
    home = tmp_path / "home"
    write_history(home, patterned_history())
    trained = run_hunch(home, "train", "--tiny", "--device", "cpu")
    assert trained.returncode == 0, trained.stderr
    champion = state_dir(home) / STATE_FILENAME
    original = champion.read_bytes()

    result = run_hunch(home, "update", "--tiny", "--device", "cpu")

    assert result.returncode == 2
    assert "no pile" in result.stderr.lower()
    assert champion.read_bytes() == original
    assert not (state_dir(home) / UPDATE_LOG_FILENAME).exists()
    counted = run_hunch(home, "stats")
    assert counted.returncode == 0, counted.stderr
    assert counted.stdout == (
        "training runs: 1\n"
        "suggestions displayed: 0\n"
        "suggestions inserted: 0\n"
    )


def test_update_discards_when_scoreboard_accuracy_falls(tmp_path: Path) -> None:
    home = tmp_path / "home"
    old = [f"old-{index}" for index in range(12)] + list(CONTEXT)
    pile = ["NOPE", *[f"new-{index}" for index in range(7)]]
    champion = install_champion(home, old, pile)
    save_scoreboard(
        state_dir(home), [Example(CONTEXT, CHAMPION_SUGGESTION)]
    )
    original = champion.read_bytes()

    result = run_hunch(home, "update", "--tiny", "--device", "cpu")

    assert result.returncode == 0, result.stderr
    assert champion.read_bytes() == original
    record = (state_dir(home) / UPDATE_LOG_FILENAME).read_text(encoding="utf-8")
    assert "discard" in record
    assert "command-ngram exact-command accuracy:" in record
    counted = run_hunch(home, "stats")
    assert counted.returncode == 0, counted.stderr
    assert "last update: discard" in counted.stdout
    assert "command-ngram exact-command accuracy:" in counted.stdout
    assert counted.stdout.startswith(
        "training runs: 0\n"
        "suggestions displayed: 0\n"
        "suggestions inserted: 0\n"
    )
    write_history(home, list(CONTEXT))
    predicted = run_hunch(home, "predict")
    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stdout == f"{CHAMPION_SUGGESTION}\n"


def test_update_keeps_when_scoreboard_accuracy_does_not_fall(tmp_path: Path) -> None:
    home = tmp_path / "home"
    old = [f"old-{index}" for index in range(12)] + list(CONTEXT)
    pile = [f"pile-{index}" for index in range(8)]
    champion = install_champion(home, old, pile)
    save_scoreboard(state_dir(home), [Example(("zz", "yy", "xx"), "never")])
    original = champion.read_bytes()

    result = run_hunch(home, "update", "--tiny", "--device", "cpu")

    assert result.returncode == 0, result.stderr
    assert champion.read_bytes() != original
    record = (state_dir(home) / UPDATE_LOG_FILENAME).read_text(encoding="utf-8")
    assert "keep" in record
    assert "command-ngram exact-command accuracy:" in record
    counted = run_hunch(home, "stats")
    assert counted.returncode == 0, counted.stderr
    assert "last update: keep" in counted.stdout
    assert "command-ngram exact-command accuracy:" in counted.stdout
    write_history(home, list(CONTEXT))
    predicted = run_hunch(home, "predict")
    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stderr == ""
    assert predicted.stdout.count("\n") <= 1
    assert "command-ngram" not in predicted.stdout
    assert SCOREBOARD_FILENAME not in predicted.stdout
    if (state_dir(home) / STATS_FILENAME).is_file():
        stored_stats = (state_dir(home) / STATS_FILENAME).read_text(encoding="utf-8")
        assert "alpha" not in stored_stats
        assert "pile-0" not in stored_stats
