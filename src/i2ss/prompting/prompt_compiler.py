from __future__ import annotations

from typing import Any, Dict, List


CONSTRAINT = "realistic, high quality field recording"

_NO_SOUND_MARKERS = (
    "no sound",
    "no sounds",
    "produces no sound",
    "produces no sounds",
    "silent",
    "silence",
    "inaudible",
)

_FALLBACK_BY_LABEL = {
    "person": "footsteps on pavement",
    "car": "car engine idling and passing traffic",
    "bicycle": "bicycle chain and wheels rolling",
    "dog": "dog barking",
}


def _append_constraint(text: str) -> str:
    base = text.strip()
    augmented = f"{base}, {CONSTRAINT}" if base else CONSTRAINT
    words = augmented.split()
    if len(words) <= 30:
        return augmented
    # Trim overly long prompts: keep the first half (max 15 words) of the base, then append constraint.
    base_words = base.split()
    head = " ".join(base_words[:15]) if base_words else ""
    head = head.strip().rstrip(",")
    return f"{head}, {CONSTRAINT}" if head else CONSTRAINT


def _sanitize_sound_prompt(text: str, label: str) -> str:
    base = (text or "").strip()
    lowered = base.lower()
    if any(marker in lowered for marker in _NO_SOUND_MARKERS) or not base:
        fallback = _FALLBACK_BY_LABEL.get((label or "").strip().lower(), f"{label or 'object'} sound")
        return fallback
    return base

def compile_for_audioldm2(vlm_json: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process VLM outputs into prompts compatible with AudioLDM2 generate command."""

    background = str(vlm_json.get("background_prompt", ""))
    compiled: Dict[str, Any] = dict(vlm_json)
    compiled["background_prompt"] = _append_constraint(background)

    objects = vlm_json.get("objects", []) or []
    processed_objects: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        augmented = dict(obj)
        label = str(obj.get("label", "object"))
        raw_sound = str(obj.get("sound_prompt", ""))
        sanitized = _sanitize_sound_prompt(raw_sound, label)
        augmented["sound_prompt"] = _append_constraint(sanitized)
        processed_objects.append(augmented)

    # Keep per-object prompts intact. Avoid "deduplication" strategies that downgrade prompts
    # to generic labels (e.g., "car sound effect"), which hurts prompt adherence.
    compiled["objects"] = processed_objects
    return compiled