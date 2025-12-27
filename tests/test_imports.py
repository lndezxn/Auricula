"""Smoke tests that ensure package modules can be imported."""

from __future__ import annotations

import importlib


def test_core_modules_importable() -> None:
    modules = ["i2ss", "i2ss.cli", "i2ss.audio", "i2ss.vision"]
    for module in modules:
        importlib.import_module(module)
