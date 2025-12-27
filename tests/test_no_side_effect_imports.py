"""Guard against importing subpackages when loading i2ss."""

from __future__ import annotations

import importlib
import sys


def test_import_i2ss_does_not_preload_cli() -> None:
    for module in ("i2ss", "i2ss.cli", "i2ss.audio", "i2ss.vision"):
        sys.modules.pop(module, None)

    importlib.import_module("i2ss")
    assert "i2ss.cli" not in sys.modules
