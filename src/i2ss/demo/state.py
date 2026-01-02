from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DemoState:
    work_dir: Path = Path("out_gradio")
    caption: str | None = None
    scene_hint: str | None = None
    seconds: int = 10
    seed: int = 0
    device: str = "cpu"
    sam_device: str = "cpu"
    audio_device: str = "auto"
    selected_id: int | None = None
    selected_overlay_path: Path | None = None
    segments_context_path: Path | None = None
    last_mix_path: Path | None = None
    objects: list[dict[str, Any]] = field(default_factory=list)
    masks: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def segments_dir(self) -> Path:
        return self.work_dir / "segments"

    @property
    def tracks_dir(self) -> Path:
        return self.work_dir / "tracks"

    @property
    def mix_dir(self) -> Path:
        return self.work_dir / "mix"
