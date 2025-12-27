"""Fast CLI smoke tests that exercise the help text via subprocess."""

from __future__ import annotations

import subprocess


def test_cli_help_commands(subprocess_env: dict[str, str]) -> None:
    cases = [
        (["python", "-m", "i2ss.cli", "--help"], "i2ss.cli"),
        (["python", "-m", "i2ss.cli", "segment", "--help"], "segment"),
        (["python", "-m", "i2ss.cli", "generate", "--help"], "generate"),
        (["python", "-m", "i2ss.cli", "mix", "--help"], "mix"),
        (["python", "-m", "i2ss.cli", "run", "--help"], "run"),
    ]

    for cmd, keyword in cases:
        result = subprocess.run(cmd, env=subprocess_env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        stdout = result.stdout.lower()
        assert keyword in stdout, "{} not mentioned in help".format(keyword)
