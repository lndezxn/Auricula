from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple


@dataclass
class ScenePrompt:
    """Represents the scene-level prompts returned by a VLM."""

    scene_caption: str
    background_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_caption": self.scene_caption,
            "background_prompt": self.background_prompt,
        }


@dataclass
class ObjectPrompt:
    """Structured result for an individual object crop."""

    id: int
    label: str
    caption: str
    sound_prompt: str
    seconds: int | None = None
    area: int | None = None
    centroid_x: float | None = None
    centroid_y: float | None = None
    box_xyxy: List[float] | None = None
    mask_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "caption": self.caption,
            "sound_prompt": self.sound_prompt,
        }
        if self.seconds is not None:
            payload["seconds"] = self.seconds
        if self.area is not None:
            payload["area"] = int(self.area)
        if self.centroid_x is not None:
            payload["centroid_x"] = float(self.centroid_x)
        if self.centroid_y is not None:
            payload["centroid_y"] = float(self.centroid_y)
        if self.box_xyxy is not None:
            payload["box_xyxy"] = list(self.box_xyxy)
        if self.mask_path is not None:
            payload["mask_path"] = self.mask_path
        return payload


def normalize_objects(entries: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize segmentation JSON objects for downstream prompt building."""

    normalized: list[dict[str, Any]] = []
    if entries is None:
        return normalized
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", "object"))
        normalized.append(
            {
                "id": entry.get("id", idx),
                "label": label,
                "box_xyxy": list(entry.get("box_xyxy", [])),
                "mask_path": entry.get("mask_path"),
                "area": int(entry.get("area", 0)),
                "centroid_x": float(entry.get("centroid_x", 0.5)),
                "centroid_y": float(entry.get("centroid_y", 0.5)),
            }
        )
    return normalized


def compute_object_seconds(area: int, min_area: int, max_area: int) -> int:
    """Map object area to a short duration window for synthesis."""

    if max_area > min_area:
        ratio = (area - min_area) / float(max_area - min_area)
    else:
        ratio = 0.5
    seconds_value = 3.0 + ratio * 2.0
    seconds_value = max(3.0, min(5.0, seconds_value))
    return int(round(seconds_value))


def object_area_range(objects: Iterable[dict[str, Any]]) -> Tuple[int, int]:
    """Return (min_area, max_area) from the iterable, falling back to (0, 0)."""

    areas = [int(max(0, obj.get("area", 0))) for obj in objects if isinstance(obj, dict)]
    if not areas:
        return 0, 0
    return min(areas), max(areas)
