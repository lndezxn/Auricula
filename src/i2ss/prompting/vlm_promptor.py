from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:  # Optional runtime dependency
    from qwen_vl_utils import process_vision_info
except ImportError as _exc:  # pragma: no cover - optional dependency guard
    process_vision_info = None  # type: ignore
    _QWEN_IMPORT_ERROR = _exc
else:
    _QWEN_IMPORT_ERROR = None

from ..utils import io as io_utils
from .scene_object_prompt import (
    ObjectPrompt,
    ScenePrompt,
    compute_object_seconds,
    normalize_objects,
    object_area_range,
)

DEFAULT_VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_VLM_DEVICE = "cuda:1"
DEFAULT_MAX_NEW_TOKENS = 256


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """Resize while keeping aspect ratio so that the longest side <= max_side."""

    width, height = image.size
    current_max = max(width, height)
    if current_max <= max_side:
        return image
    scale = max_side / float(current_max)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.BICUBIC)


@dataclass
class _VLMResult:
    raw_text: str
    payload: dict[str, Any]


class VLMPromptor:
    """Generate scene and object prompts via a vision-language model."""

    def __init__(
        self,
        model_id: str = DEFAULT_VLM_MODEL,
        device: str = DEFAULT_VLM_DEVICE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        if process_vision_info is None:
            raise ImportError(
                "qwen-vl-utils is required for VLMPromptor; install qwen-vl-utils and retry"
            ) from _QWEN_IMPORT_ERROR
        self.model_id = model_id
        self.device = device
        dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map={"": device},
        )
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        image_path: Path | str,
        objects_json_path: Path | str,
        seconds: int = 10,
        use_cache: bool = True,
        cache_dir: Path | str | None = None,
        debug_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        objects_json_path = Path(objects_json_path)
        cache_path = self._resolve_cache_path(image_path, objects_json_path, cache_dir)
        if use_cache and cache_path is not None and cache_path.exists():
            with cache_path.open(encoding="utf-8") as fp:
                return json.load(fp)

        scene_objects = self._load_objects(objects_json_path)
        base_image = Image.open(image_path).convert("RGB")
        scene_prompt = self._generate_scene_prompt(base_image, seconds, debug_dir)
        min_area, max_area = object_area_range(scene_objects)
        object_prompts: list[dict[str, Any]] = []
        for obj in scene_objects:
            crop = self._crop_object(base_image, objects_json_path.parent, obj)
            object_result = self._generate_object_prompt(crop, obj.get("label", "object"), debug_dir)
            seconds_value = compute_object_seconds(int(obj.get("area", 0)), min_area, max_area)
            normalized = ObjectPrompt(
                id=int(obj.get("id", len(object_prompts))),
                label=str(obj.get("label", "object")),
                caption=object_result.get("caption", ""),
                sound_prompt=object_result.get("sound_prompt", ""),
                seconds=seconds_value,
                area=int(obj.get("area", 0)),
                centroid_x=float(obj.get("centroid_x", 0.5)),
                centroid_y=float(obj.get("centroid_y", 0.5)),
                box_xyxy=list(obj.get("box_xyxy", [])),
                mask_path=obj.get("mask_path"),
            ).to_dict()
            object_prompts.append(normalized)

        prompts: dict[str, Any] = {
            "seconds": int(seconds),
            "scene_caption": scene_prompt.scene_caption,
            "background_prompt": scene_prompt.background_prompt,
            "objects": object_prompts,
        }
        if cache_path is not None:
            io_utils.ensure_dir(cache_path.parent)
            io_utils.write_json(cache_path, prompts)
        return prompts

    def _generate_scene_prompt(
        self,
        image: Image.Image,
        seconds: int,
        debug_dir: Path | str | None,
    ) -> ScenePrompt:
        instruction = (
            "You are an audio prompt engineer. "
            "Return a strict JSON object with keys scene_caption and background_prompt. "
            "scene_caption: 1-2 sentences describing the entire image in English. "
            "background_prompt: a concise ambient sound bed description for a realistic field recording, "
            "lasting about {seconds} seconds, no music, no singing, no narration. "
            "Do not include any extra text outside the JSON object."
        ).format(seconds=seconds)
        result = self._invoke_vlm(
            messages=[
                {
                    "role": "user",
                    "content": [
                                {"type": "image", "image": resize_max_side(image, 896)},
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
            debug_path=self._debug_path(debug_dir, "scene.txt"),
        )
        scene_caption = str(result.payload.get("scene_caption", "")).strip()
        background_prompt = str(result.payload.get("background_prompt", "")).strip()
        if not background_prompt:
            background_prompt = (
                "ambient environmental soundscape, realistic field recording, no music, no narration, {seconds} seconds"
            ).format(seconds=seconds)
        return ScenePrompt(scene_caption=scene_caption, background_prompt=background_prompt)

    def _generate_object_prompt(
        self,
        crop: Image.Image,
        label: str,
        debug_dir: Path | str | None,
    ) -> dict[str, str]:
        instruction = (
            "You see a cropped region of a segmented object labeled '{label}'. "
            "Respond with a strict JSON object: {{\"caption\": <visual_description>, \"sound_prompt\": <sound_description>}}. "
            "caption: concise English description of this object in the crop. "
            "sound_prompt: one sentence (<= 22 words) in English describing a realistic sound effect or field recording of this object; "
            "no music, no singing, no narration. Return JSON only."
        ).format(label=label)
        result = self._invoke_vlm(
            messages=[
                {
                    "role": "user",
                    "content": [
                                {"type": "image", "image": resize_max_side(crop, 672)},
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
            debug_path=self._debug_path(debug_dir, f"object_{label}.txt"),
        )
        caption = str(result.payload.get("caption", "")).strip()
        sound_prompt = str(result.payload.get("sound_prompt", "")).strip()
        return {"caption": caption, "sound_prompt": sound_prompt}

    def _invoke_vlm(self, messages: Sequence[dict[str, Any]], debug_path: Path | None) -> _VLMResult:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        vision_inputs = process_vision_info(messages)
        if isinstance(vision_inputs, tuple):  # backward compatibility with tuple return
            images, videos = vision_inputs
        else:
            images = vision_inputs.get("images")  # type: ignore[arg-type]
            videos = vision_inputs.get("videos") if isinstance(vision_inputs, dict) else None
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        raw_text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        payload = self._parse_json_output(raw_text, debug_path)
        return _VLMResult(raw_text=raw_text, payload=payload)

    def _parse_json_output(self, raw_text: str, debug_path: Path | None) -> dict[str, Any]:
        # Prefer text after the assistant role marker to avoid braces inside instructions.
        lowered = raw_text.lower()
        role_pos = lowered.rfind("assistant")
        search_space = raw_text[role_pos:] if role_pos != -1 else raw_text
        start = search_space.find("{")
        end = search_space.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = search_space[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        if debug_path is not None:
            io_utils.write_text(debug_path, raw_text)
        raise ValueError("Model output is not valid JSON")

    def _load_objects(self, objects_json_path: Path) -> list[dict[str, Any]]:
        if not objects_json_path.exists():
            raise FileNotFoundError(f"Missing objects JSON at {objects_json_path}")
        with objects_json_path.open(encoding="utf-8") as fp:
            payload = json.load(fp)
        entries = payload.get("objects", []) if isinstance(payload, dict) else []
        return normalize_objects(entries)

    def _crop_object(self, image: Image.Image, base_dir: Path, obj: dict[str, Any]) -> Image.Image:
        width, height = image.size
        box = obj.get("box_xyxy") or [0.0, 0.0, float(width), float(height)]
        x1, y1, x2, y2 = self._normalize_box(box, width, height)
        crop = image.crop((x1, y1, x2, y2))
        mask_path = obj.get("mask_path")
        if mask_path:
            mask_full_path = base_dir / mask_path
            if mask_full_path.exists():
                mask = cv2.imread(str(mask_full_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_crop = mask[y1:y2, x1:x2]
                    crop = self._overlay_mask_outline(crop, mask_crop)
        return crop

    def _normalize_box(self, box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
        if len(box) != 4:
            return 0, 0, width, height
        x1, y1, x2, y2 = [float(v) for v in box]
        w = x2 - x1
        h = y2 - y1
        padding = 0.15 * max(w, h)
        x1p = max(0, int(math.floor(x1 - padding)))
        y1p = max(0, int(math.floor(y1 - padding)))
        x2p = min(width, int(math.ceil(x2 + padding)))
        y2p = min(height, int(math.ceil(y2 + padding)))
        return x1p, y1p, x2p, y2p

    def _overlay_mask_outline(self, crop: Image.Image, mask: np.ndarray) -> Image.Image:
        if mask is None or mask.size == 0:
            return crop
        mask_uint8 = np.asarray(mask, dtype=np.uint8)
        if mask_uint8.ndim > 2:
            mask_uint8 = mask_uint8[:, :, 0]
        _, thresh = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = np.array(crop.convert("RGB"))
        if contours:
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=3)
        return Image.fromarray(overlay)

    def _resolve_cache_path(
        self,
        image_path: Path,
        objects_json_path: Path,
        cache_dir: Path | str | None,
    ) -> Path | None:
        if cache_dir is None:
            return None
        cache_dir = Path(cache_dir)
        digest = self._hash_inputs(image_path, objects_json_path)
        return cache_dir / f"vlm_prompts_{digest}.json"

    def _hash_inputs(self, image_path: Path, objects_json_path: Path) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(image_path.resolve()).encode("utf-8"))
        hasher.update(str(objects_json_path.resolve()).encode("utf-8"))
        if objects_json_path.exists():
            hasher.update(objects_json_path.read_bytes())
        hasher.update(self.model_id.encode("utf-8"))
        return hasher.hexdigest()[:16]

    def _debug_path(self, debug_dir: Path | str | None, name: str) -> Path | None:
        if debug_dir is None:
            return None
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir / name
