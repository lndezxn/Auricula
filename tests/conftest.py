from __future__ import annotations

import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Callable, Dict

import jsonschema
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _png_chunk(name: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + name
        + data
        + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def src_root() -> Path:
    return SRC_ROOT


@pytest.fixture
def subprocess_env(src_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    path_value = str(src_root)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{path_value}{os.pathsep}{existing}" if existing else path_value
    )
    return env


@pytest.fixture
def make_dummy_image() -> Callable[[Path | str, int, int], Path]:
    def _make(path: Path | str, width: int, height: int) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")

        raw = bytearray()
        for y in range(height):
            raw.append(0)  # use "no filter" per PNG row
            for x in range(width):
                base = (x * 3 + y * 5) % 256
                raw.extend((base, (base + 64) % 256, (base + 128) % 256))

        compressor = zlib.compressobj()
        compressed = compressor.compress(bytes(raw)) + compressor.flush()
        png_bytes = b"\x89PNG\r\n\x1a\n"
        png_bytes += _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        png_bytes += _png_chunk(b"IDAT", compressed)
        png_bytes += _png_chunk(b"IEND", b"")
        path.write_bytes(png_bytes)
        return path

    return _make


@pytest.fixture
def assert_wav_ok() -> Callable[[Path | str, float, int | None, bool | None], None]:
    def _assert(
        path: Path | str,
        min_seconds: float,
        sr: int | None = None,
        stereo: bool | None = None,
    ) -> None:
        path = Path(path)
        if not path.exists():
            raise AssertionError(f"{path} missing")

        data, samplerate = sf.read(str(path))
        frames = data.shape[0]
        duration = frames / samplerate
        if sr is not None:
            assert samplerate == sr, "unexpected sample rate"
        if stereo is not None:
            channels = 1 if data.ndim == 1 else data.shape[1]
            expected = 2 if stereo else 1
            assert channels == expected, "unexpected channel count"
        assert duration >= float(min_seconds), "file too short"

    return _assert


@pytest.fixture
def assert_json_schema() -> Callable[[Any, dict[str, Any]], None]:
    def _assert(obj: Any, schema: dict[str, Any]) -> None:
        jsonschema.validate(instance=obj, schema=schema)

    return _assert
