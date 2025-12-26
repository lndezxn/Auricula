"""Vision helpers for building segmentation overlays."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def describe_queries(queries: Sequence[str]) -> dict[str, Any]:
    """Minimal metadata describing what the queries represent."""
    return {
        "objects": list(queries),
        "object_count": len(queries),
    }


def create_overlay(path: Path, description: str) -> None:
    """Serialize a simple overlay description."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(description, encoding="utf-8")
