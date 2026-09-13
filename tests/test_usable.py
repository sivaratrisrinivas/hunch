from __future__ import annotations

from pathlib import Path
import subprocess

from hunch.history import (
    HistoryLine,
    awk_count_usable,
    classify_history_line,
    is_sensitive,
    is_usable,
)


SENSITIVE_FIXTURES = [
    "export API_TOKEN=super-secret-value",
    "curl -H 'Authorization: Bearer abc123' https://example.test",
    "ssh -i ~/.ssh/id_rsa server",
    "https://admin:hunter2@example.test/private",
    "PGPASSWORD=hunter2 psql",
    "export AWS_SECRET_ACCESS_KEY=hunter2",
    "API-KEY=hunter2 deploy",
    "token:hunter2 command",
]

CORPUS = [
    "git status",
    "echo token",
    *SENSITIVE_FIXTURES,
    "",
    "#1710000000",
    "   ",
]


def test_classify_and_is_usable_agree_on_the_corpus() -> None:
    expected = {
        "git status": HistoryLine.USABLE,
        "echo token": HistoryLine.USABLE,
        "": HistoryLine.EMPTY,
        "#1710000000": HistoryLine.TIMESTAMP,
        "   ": HistoryLine.EMPTY,
    }
    for command in SENSITIVE_FIXTURES:
        expected[command] = HistoryLine.SENSITIVE

    for line in CORPUS:
        kind = classify_history_line(line)
        assert kind is expected[line]
        assert is_usable(line) is (kind is HistoryLine.USABLE)
        assert is_sensitive(line) is (kind is HistoryLine.SENSITIVE)


def test_echo_token_is_usable_and_url_userinfo_is_not() -> None:
    assert is_usable("echo token")
    assert not is_sensitive("echo token")
    assert not is_usable("https://admin:hunter2@example.test/private")
    assert is_sensitive("https://admin:hunter2@example.test/private")


def test_awk_count_usable_agrees_with_python_on_a_histfile(tmp_path: Path) -> None:
    histfile = tmp_path / "histfile"
    histfile.write_text("\n".join(CORPUS) + "\n", encoding="utf-8")
    counted = subprocess.run(
        ["awk", "-f", "/dev/stdin", str(histfile)],
        input=awk_count_usable(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert counted.returncode == 0, counted.stderr
    assert int(counted.stdout.strip()) == sum(is_usable(line) for line in CORPUS)
