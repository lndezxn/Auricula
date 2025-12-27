"""Verify CLI honors the output layout without downloading external artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_run_populates_layout(
    tmp_path: Path,
    make_dummy_image,
    subprocess_env: dict[str, str],
) -> None:
    image_path = tmp_path / "sample.png"
    make_dummy_image(image_path, width=16, height=16)
    out_root = tmp_path / "out"
    cmd = [
        "python",
        "-m",
        "i2ss.cli",
        "run",
        "--image",
        str(image_path),
        "--queries",
        "car.person",
        "--out",
        str(out_root),
    ]

    result = subprocess.run(cmd, env=subprocess_env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    for sub in ("segments", "tracks", "mix"):
        assert (out_root / sub).is_dir(), "{} missing".format(sub)
