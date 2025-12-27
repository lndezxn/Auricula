from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from ..utils import audio_io

NEGATIVE_PROMPT = "music, singing, melody, instruments, distortion, low quality"
BACKGROUND_TEMPLATES: dict[str, str] = {
    "city": "cityscape ambience with layered traffic, subway hum, and distant chatter, realistic, stereo, {seconds} seconds",
    "indoor": "indoor room ambience, subtle reverb, quiet background noise, realistic, stereo, {seconds} seconds",
    "beach": "relaxing beach ambience, ocean waves, seagulls, realistic, stereo, {seconds} seconds",
    "forest": "forest ambience with birds and rustling leaves, gentle breeze, realistic, stereo, {seconds} seconds",
    "rain": "rainy day ambience with steady rain, distant thunder, realistic, stereo, {seconds} seconds",
    "snow": "snowy ambience with muffled footsteps, gentle wind, realistic, stereo, {seconds} seconds",
    "unknown": "ambient environmental ambience, distant soundscape, realistic, stereo, {seconds} seconds",
}

SOUND_PHRASES: dict[str, str] = {
    "dog": "dog barking",
    "car": "car engine and passing traffic",
    "person": "people talking and footsteps",
    "bird": "birds chirping",
    "waves": "ocean waves",
    "ocean": "ocean waves",
    "rain": "rainfall",
    "engine": "engine idling and humming",
    "footsteps": "footsteps on pavement",
    "train": "train horn and rumble",
    "motorcycle": "motorcycle revving",
    "bus": "bus engine idling and doors opening",
    "truck": "truck rumbling and braking",
    "keyboard": "typing on a keyboard",
    "door": "door opening and closing",
    "child": "children laughing and playing",
    "crowd": "crowd chatter and footsteps",
}

SCENE_KEYWORDS = [
    ("beach", ("beach", "ocean", "wave", "sea", "sand")),
    ("forest", ("forest", "tree", "woods", "park")),
    ("indoor", ("kitchen", "room", "office", "bedroom", "indoor", "studio")),
    ("street", ("street", "road", "traffic", "car", "sidewalk", "crosswalk")),
    ("city", ("city", "downtown", "urban", "skyscraper")),
    ("crowd", ("crowd", "people", "market", "fair", "festival")),
]

TIME_KEYWORDS = {
    "night": ("night", "dark", "evening", "dusk", "streetlight"),
    "day": ("day", "sunny", "morning", "afternoon", "bright"),
}

WEATHER_KEYWORDS = {
    "rain": ("rain", "umbrella", "storm", "shower"),
    "snow": ("snow", "blizzard", "white", "frost"),
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "with",
    "in",
    "on",
    "at",
    "from",
    "this",
    "that",
    "scene",
    "image",
    "photo",
    "showing",
}


def infer_scene_tags(caption: str, labels: list[str]) -> dict[str, Any]:
    buffer = caption.lower() if caption else ""
    label_tokens = " ".join(label.lower() for label in labels if label)
    text = f"{buffer} {label_tokens}".strip()
    keywords = re.findall(r"\b\w+\b", text)
    detected_scene = "unknown"
    for scene, candidates in SCENE_KEYWORDS:
        if any(candidate in text for candidate in candidates):
            detected_scene = scene
            break
    time_tag = "unknown"
    for tag, candidates in TIME_KEYWORDS.items():
        if any(candidate in text for candidate in candidates):
            time_tag = tag
            break
    weather: list[str] = []
    for weather_tag, candidates in WEATHER_KEYWORDS.items():
        if any(candidate in text for candidate in candidates):
            weather.append(weather_tag)
    unique_keywords: list[str] = []
    for token in keywords:
        if token in STOPWORDS or len(token) <= 2:
            continue
        if token not in unique_keywords:
            unique_keywords.append(token)
    for label in labels:
        normalized = label.lower().strip()
        if normalized and normalized not in unique_keywords:
            unique_keywords.append(normalized)
    scene_tags = {
        "scene": detected_scene,
        "time": time_tag,
        "keywords": unique_keywords,
    }
    if weather:
        scene_tags["weather"] = weather
    return scene_tags

DEFAULT_SAMPLE_RATE = 16000
BACKGROUND_GAIN_DB = -8.0
OBJECT_GAIN_SOURCE_MIN_DB = -14.0
OBJECT_GAIN_SOURCE_MAX_DB = -6.0
OBJECT_GAIN_CLAMP_MIN_DB = -18.0
OBJECT_GAIN_CLAMP_MAX_DB = -4.0
MIXING_CONFIG = {
    "peak_db": -1.0,
    "reverb_preset": "outdoor_light",
    "limiter": True,
}



def _load_json(source: str | Path) -> dict[str, Any]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _normalize_objects(entries: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries or []):
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", f"object_{idx}"))
        centroid_x = float(entry.get("centroid_x", 0.5))
        centroid_y = float(entry.get("centroid_y", 0.5))
        area = int(max(0, entry.get("area", 0)))
        normalized.append(
            {
                "id": entry.get("id", idx),
                "label": label,
                "area": area,
                "centroid_x": max(0.0, min(1.0, centroid_x)),
                "centroid_y": max(0.0, min(1.0, centroid_y)),
            }
        )
    return normalized


def load_objects(objects_json_path: str | Path) -> list[dict[str, Any]]:
    context = _load_json(objects_json_path)
    entries = context.get("objects", [])
    return _normalize_objects(entries)


def load_segments_context(segments_json_path: str | Path) -> dict[str, Any]:
    payload = _load_json(segments_json_path)
    scene_tags = payload.get("scene_tags")
    if not isinstance(scene_tags, dict):
        scene_tags = {"scene": "unknown"}
    return {
        "image_path": str(payload.get("image_path", segments_json_path)),
        "caption": str(payload.get("caption", "")),
        "scene_tags": scene_tags,
        "objects": _normalize_objects(payload.get("objects", [])),
    }



def _background_template(scene: str, caption_hint: str, seconds: int) -> str:
    scene_key = (scene or "").lower()
    caption_lower = caption_hint.lower()
    if scene_key == "street":
        if any(token in caption_lower for token in ("dirt road", "road", "dust")):
            base = "rural dirt road ambience, wind, distant vehicle noise, realistic, stereo, {seconds} seconds"
        else:
            base = "busy city street ambience, distant traffic, occasional honks, realistic, stereo, {seconds} seconds"
    else:
        base = BACKGROUND_TEMPLATES.get(scene_key, BACKGROUND_TEMPLATES["unknown"])
    prompt = base.format(seconds=seconds)
    if caption_hint:
        prompt = f"{prompt} Scene note: {caption_hint}."
    return prompt


def sound_phrase_for_label(label: str) -> str:
    candidate = label.lower()
    for key, phrase in SOUND_PHRASES.items():
        if key in candidate:
            return phrase
    return f"{label} sound effect"


def _compute_object_seconds(area: int, min_area: int, max_area: int) -> int:
    if max_area > min_area:
        ratio = (area - min_area) / float(max_area - min_area)
    else:
        ratio = 0.5
    seconds_value = 3.0 + ratio * 2.0
    seconds_value = max(3.0, min(5.0, seconds_value))
    return int(round(seconds_value))


def _compute_gain_db(area: int, a_min: float, a_max: float) -> float:
    a_value = math.log10(area + 1)
    delta = (a_max - a_min) or 0.0
    if delta < 1e-6:
        t = 0.5
    else:
        t = (a_value - a_min) / delta
    t = max(0.0, min(1.0, t))
    gain = OBJECT_GAIN_SOURCE_MIN_DB + (OBJECT_GAIN_SOURCE_MAX_DB - OBJECT_GAIN_SOURCE_MIN_DB) * t
    clipped = max(OBJECT_GAIN_CLAMP_MIN_DB, min(OBJECT_GAIN_CLAMP_MAX_DB, gain))
    return round(clipped, 2)


def build_audio_prompts(
    segments_context_json_path: str | Path,
    seconds: int = 10,
    seed: int = 0,
    scene_hint: str | None = None,
) -> dict[str, Any]:
    normalized_seconds = max(1, seconds)
    context = load_segments_context(segments_context_json_path)
    caption_text = context["caption"]
    scene_tags = context["scene_tags"]
    objects = context["objects"]
    prompt_caption = " ".join(filter(None, [caption_text, scene_hint or ""]))
    background_prompt = _background_template(scene_tags.get("scene", "unknown"), prompt_caption, normalized_seconds)
    area_values = [obj["area"] for obj in objects]
    min_area = min(area_values) if area_values else 0
    max_area = max(area_values) if area_values else 0
    object_entries: list[dict[str, Any]] = []
    reference_caption = prompt_caption or caption_text or "the scene"
    for obj in objects:
        obj_seconds = _compute_object_seconds(obj["area"], min_area, max_area)
        object_prompt = ", ".join(
            [
                sound_phrase_for_label(obj["label"]),
                f"in the scene: {reference_caption}",
                "realistic sound effect",
                "high quality",
                "stereo",
                f"{obj_seconds} seconds",
            ]
        )
        object_entries.append(
            {
                "id": obj["id"],
                "label": obj["label"],
                "centroid_x": obj["centroid_x"],
                "centroid_y": obj["centroid_y"],
                "area": obj["area"],
                "seconds": obj_seconds,
                "prompt": object_prompt,
                "negative_prompt": NEGATIVE_PROMPT,
            }
        )
    prompts: dict[str, Any] = {
        "image_path": context["image_path"],
        "seconds": normalized_seconds,
        "seed": seed,
        "caption": caption_text,
        "scene_tags": scene_tags,
        "background": {
            "prompt": background_prompt,
            "negative_prompt": NEGATIVE_PROMPT,
        },
        "objects": object_entries,
    }
    return prompts


def build_mix_meta(prompts: dict[str, Any], tracks_dir: Path) -> dict[str, Any]:
    objects = prompts.get("objects", [])
    logs = [math.log10(obj.get("area", 0) + 1) for obj in objects]
    a_min = min(logs) if logs else 0.0
    a_max = max(logs) if logs else 0.0
    meta_objects: list[dict[str, Any]] = []
    for obj in objects:
        centroid_x = float(obj.get("centroid_x", 0.5))
        area = int(obj.get("area", 0))
        pan = max(-1.0, min(1.0, (centroid_x - 0.5) * 2.0))
        gain_db = _compute_gain_db(area, a_min, a_max)
        path = audio_io.build_object_path(tracks_dir, obj.get("id"), obj.get("label", "object"))
        meta_objects.append(
            {
                "id": obj.get("id"),
                "label": obj.get("label"),
                "path": str(path),
                "centroid_x": centroid_x,
                "centroid_y": float(obj.get("centroid_y", 0.5)),
                "area": area,
                "pan": round(pan, 3),
                "gain_db": gain_db,
            }
        )
    meta: dict[str, Any] = {
        "image_path": prompts.get("image_path"),
        "seconds": prompts.get("seconds"),
        "seed": prompts.get("seed"),
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "background": {
            "path": str(tracks_dir / "background.wav"),
            "gain_db": BACKGROUND_GAIN_DB,
        },
        "objects": meta_objects,
        "mixing": MIXING_CONFIG,
    }
    return meta


def save_prompts(out_dir: str | Path, prompts: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = out_dir / "prompts.json"
    with prompts_path.open("w", encoding="utf-8") as fp:
        json.dump(prompts, fp, ensure_ascii=False, indent=2)


def save_mix_meta(out_dir: str | Path, meta: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
