from __future__ import annotations

import difflib
from typing import Any, Dict, List


CONSTRAINT = "realistic, no music, no speech, high quality field recording"


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


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _deduplicate_objects(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for obj in objs:
        sound = str(obj.get("sound_prompt", ""))
        label = str(obj.get("label", "object"))
        area = int(obj.get("area", 0))
        duplicate_of = None
        for idx, existing in enumerate(kept):
            if _similar(sound, str(existing.get("sound_prompt", ""))) > 0.85:
                duplicate_of = idx
                break
        if duplicate_of is None:
            kept.append(obj)
            continue
        # Decide which to keep based on area
        existing = kept[duplicate_of]
        existing_area = int(existing.get("area", 0))
        if area > existing_area:
            # Replace existing with the larger area version; downgrade existing
            downgraded = {
                **existing,
                "sound_prompt": _append_constraint(f"{existing.get('label', 'object')} sound effect"),
            }
            kept[duplicate_of] = obj
            kept.append(downgraded)
        else:
            downgraded = {
                **obj,
                "sound_prompt": _append_constraint(f"{label} sound effect"),
            }
            kept.append(downgraded)
    return kept


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
        augmented["sound_prompt"] = _append_constraint(str(obj.get("sound_prompt", "")))
        processed_objects.append(augmented)

    compiled["objects"] = _deduplicate_objects(processed_objects)
    return compiled