from __future__ import annotations

import json
from pathlib import Path
import stat

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


def test_update_waits_when_cuda_is_hidden_and_still_feeds_command_ngram(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    old = [f"old-{index}" for index in range(12)] + list(CONTEXT)
    pile = [f"pile-{index}" for index in range(8)]
    champion = install_champion(home, old, pile)
    save_scoreboard(
        state_dir(home), [Example(CONTEXT, CHAMPION_SUGGESTION)]
    )
    original = champion.read_bytes()

    result = run_hunch(home, "update", cuda_visible_devices="")

    assert result.returncode == 0, result.stderr
    assert champion.read_bytes() == original
    record = (state_dir(home) / UPDATE_LOG_FILENAME).read_text(encoding="utf-8")
    assert "the transformer waited" in record
    assert "command-ngram exact-command accuracy:" in record
    assert "the transformer waited" in result.stdout
    assert "command-ngram exact-command accuracy:" in result.stdout
    consumed = json.loads(
        (state_dir(home) / CONSUMED_FILENAME).read_text(encoding="utf-8")
    )
    assert consumed == {"consumed": len(old) + 8}
    counted = run_hunch(home, "stats")
    assert counted.returncode == 0, counted.stderr
    assert "last update: the transformer waited" in counted.stdout
    assert "command-ngram exact-command accuracy:" in counted.stdout
    write_history(home, list(CONTEXT))
    predicted = run_hunch(home, "predict", cuda_visible_devices="")
    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stdout == f"{CHAMPION_SUGGESTION}\n"
    again = run_hunch(home, "update", cuda_visible_devices="")
    assert again.returncode == 2
    assert "no pile" in again.stderr.lower()
    assert champion.read_bytes() == original


def test_exam_before_update_fails_and_does_not_change_the_champion(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    old = [f"old-{index}" for index in range(12)] + list(CONTEXT)
    pile = [f"pile-{index}" for index in range(8)]
    champion = install_champion(home, old, pile)
    original = champion.read_bytes()

    result = run_hunch(home, "exam")

    assert result.returncode == 2
    assert "exam" in result.stderr.lower()
    assert "update" in result.stderr.lower()
    assert result.stdout == ""
    assert champion.read_bytes() == original
    assert not _exam_record_paths(home)


def test_exam_does_not_resit_the_scoreboard(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    result = run_hunch(home, "exam", "--now")

    assert result.returncode != 0
    assert result.stdout == ""
    assert "unrecognized arguments" in result.stderr
    assert "--now" in result.stderr


def test_exam_after_discard_names_the_lost_pair_and_leaves_the_champion(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    old = [f"old-{index}" for index in range(12)] + list(CONTEXT)
    pile = ["NOPE", *CONTEXT, "NOPE", *CONTEXT]
    champion = install_champion(home, old, pile)
    save_scoreboard(
        state_dir(home), [Example(CONTEXT, CHAMPION_SUGGESTION)]
    )
    original = champion.read_bytes()

    updated = run_hunch(home, "update", "--tiny", "--device", "cpu")
    examined = run_hunch(home, "exam")
    again = run_hunch(home, "exam")
    counted = run_hunch(home, "stats")

    assert updated.returncode == 0, updated.stderr
    assert "discard" in updated.stdout
    assert champion.read_bytes() == original
    assert examined.returncode == 0, examined.stderr
    assert examined.stderr == ""
    lost, gained, pairs = _parse_exam_stdout(examined.stdout)
    assert lost == 1
    assert gained == 0
    assert pairs == [("lost", CONTEXT, CHAMPION_SUGGESTION)]
    assert again.returncode == 0, again.stderr
    assert again.stdout == examined.stdout
    assert champion.read_bytes() == original
    records = _exam_record_paths(home)
    assert len(records) == 1
    assert stat.S_IMODE(records[0].stat().st_mode) == 0o600
    stored = records[0].read_text(encoding="utf-8")
    assert CHAMPION_SUGGESTION in stored
    assert "alpha" in stored
    assert counted.returncode == 0, counted.stderr
    assert "last update: discard" in counted.stdout
    assert "command-ngram exact-command accuracy:" in counted.stdout
    assert CHAMPION_SUGGESTION not in counted.stdout
    assert "alpha" not in counted.stdout
    if (state_dir(home) / STATS_FILENAME).is_file():
        stored_stats = (state_dir(home) / STATS_FILENAME).read_text(encoding="utf-8")
        assert CHAMPION_SUGGESTION not in stored_stats
        assert "alpha" not in stored_stats


def test_exam_after_keep_prints_counts_and_looking_leaves_the_champion(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    old = [f"old-{index}" for index in range(12)] + list(CONTEXT)
    pile = [f"pile-{index}" for index in range(8)]
    champion = install_champion(home, old, pile)
    save_scoreboard(state_dir(home), [Example(("zz", "yy", "xx"), "never")])
    before_update = champion.read_bytes()

    updated = run_hunch(home, "update", "--tiny", "--device", "cpu")
    kept = champion.read_bytes()
    examined = run_hunch(home, "exam")
    counted = run_hunch(home, "stats")

    assert updated.returncode == 0, updated.stderr
    assert "keep" in updated.stdout
    assert kept != before_update
    assert examined.returncode == 0, examined.stderr
    assert examined.stderr == ""
    lost, gained, pairs = _parse_exam_stdout(examined.stdout)
    assert lost == 0
    assert gained == 0
    assert pairs == []
    assert champion.read_bytes() == kept
    assert counted.returncode == 0, counted.stderr
    assert "last update: keep" in counted.stdout
    assert "command-ngram exact-command accuracy:" in counted.stdout
    assert "never" not in counted.stdout
    assert "pile-0" not in counted.stdout


def test_exam_after_wait_says_the_transformer_waited(tmp_path: Path) -> None:
    home = tmp_path / "home"
    old = [f"old-{index}" for index in range(12)] + list(CONTEXT)
    pile = [f"pile-{index}" for index in range(8)]
    champion = install_champion(home, old, pile)
    save_scoreboard(
        state_dir(home), [Example(CONTEXT, CHAMPION_SUGGESTION)]
    )
    original = champion.read_bytes()

    updated = run_hunch(home, "update", cuda_visible_devices="")
    examined = run_hunch(home, "exam")
    counted = run_hunch(home, "stats")

    assert updated.returncode == 0, updated.stderr
    assert "the transformer waited" in updated.stdout
    assert examined.returncode == 0, examined.stderr
    assert examined.stderr == ""
    assert examined.stdout == "the transformer waited\n"
    assert champion.read_bytes() == original
    records = _exam_record_paths(home)
    assert len(records) == 1
    assert stat.S_IMODE(records[0].stat().st_mode) == 0o600
    stored = records[0].read_text(encoding="utf-8")
    assert "waited" in stored
    assert CHAMPION_SUGGESTION not in stored
    assert "alpha" not in stored
    assert counted.returncode == 0, counted.stderr
    assert "last update: the transformer waited" in counted.stdout
    assert CHAMPION_SUGGESTION not in counted.stdout
    assert "alpha" not in counted.stdout


KNOWN_STATE_FILES = {
    CONSUMED_FILENAME,
    SCOREBOARD_FILENAME,
    STATE_FILENAME,
    STATS_FILENAME,
    UPDATE_LOG_FILENAME,
    "inspect.jsonl",
    "update.lock",
}


def _exam_record_paths(home: Path) -> list[Path]:
    directory = state_dir(home)
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name not in KNOWN_STATE_FILES
    )


def _parse_exam_stdout(
    stdout: str,
) -> tuple[int, int, list[tuple[str, tuple[str, ...], str]]]:
    lines = stdout.splitlines()
    assert len(lines) >= 2
    assert lines[0].startswith("lost ")
    assert lines[1].startswith("gained ")
    lost = int(lines[0].removeprefix("lost "))
    gained = int(lines[1].removeprefix("gained "))
    pairs: list[tuple[str, tuple[str, ...], str]] = []
    for line in lines[2:]:
        kind, body = line.split(" ", 1)
        assert kind in {"lost", "gained"}
        context_text, next_command = body.split(" -> ", 1)
        pairs.append((kind, tuple(context_text.split(" / ")), next_command))
    assert lost == sum(kind == "lost" for kind, _, _ in pairs)
    assert gained == sum(kind == "gained" for kind, _, _ in pairs)
    return lost, gained, pairs
