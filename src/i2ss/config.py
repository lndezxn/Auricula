"""Configuration dataclass and YAML helpers for the i2ss workflow."""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass
class Config:
    """Holds output layout and default runtime values."""

    out_dir: Path = Path("out")
    sampling_rate: int = 22050
    seconds: int = 10
    seed: int = 0

    segments_subdir: str = field(init=False, default="segments")
    tracks_subdir: str = field(init=False, default="tracks")
    mix_subdir: str = field(init=False, default="mix")

    segments_dir: Path = field(init=False)
    tracks_dir: Path = field(init=False)
    mix_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.segments_dir = self.out_dir / self.segments_subdir
        self.tracks_dir = self.out_dir / self.tracks_subdir
        self.mix_dir = self.out_dir / self.mix_subdir

    def ensure_dirs(self) -> None:
        """Create the base directories that the CLI expects."""
        for path in (self.out_dir, self.segments_dir, self.tracks_dir, self.mix_dir):
            path.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> Dict[str, Any]:
        """Serialize configurable fields (excludes derived paths)."""
        return {
            "out_dir": str(self.out_dir),
            "sampling_rate": self.sampling_rate,
            "seconds": self.seconds,
            "seed": self.seed,
        }

    def save(self, path: Path) -> None:
        """Persist the configuration to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(self.as_dict(), fp)

    @classmethod
    def load(
        cls,
        yaml_path: Optional[Path] = None,
        **overrides: Any,
    ) -> Config:
        """Load configuration from a YAML file and apply optional overrides."""
        values: Dict[str, Any] = {}
        if yaml_path is not None and yaml_path.exists():
            with Path(yaml_path).open(encoding="utf-8") as fp:
                content = yaml.safe_load(fp)
            if isinstance(content, Mapping):
                values.update(content)
        values.update(overrides)
        if "out_dir" in values:
            values["out_dir"] = Path(values["out_dir"])
        return cls(**values)
