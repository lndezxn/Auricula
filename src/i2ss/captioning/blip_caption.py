from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Tuple

from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


DEFAULT_MODEL_ID = "Salesforce/blip-image-captioning-large"


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    print("Captioning falling back to CPU because CUDA is unavailable.")
    return "cpu"


def load_captioner(
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
    dtype: str = "float16",
) -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    resolved_device = _resolve_device(device)
    effective_dtype = "float32" if resolved_device == "cpu" else dtype
    torch_dtype = torch.float16 if effective_dtype == "float16" else torch.float32

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(resolved_device)
    model.eval()
    return processor, model


def caption_image(
    image_path: str,
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
    dtype: str = "float16",
    max_new_tokens: int = 40,
    num_beams: int = 5,
) -> str:
    resolved_device = _resolve_device(device)
    if resolved_device == "cpu":
        dtype = "float32"
    processor, model = load_captioner(model_id=model_id, device=resolved_device, dtype=dtype)

    image = Image.open(str(image_path)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(resolved_device)
    with torch.no_grad():
        tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    caption = processor.decode(tokens[0], skip_special_tokens=True).strip()
    if not caption:
        caption = "An image with a rich scene."
    return caption


def save_caption(out_dir: str | Path, image_path: str | Path, caption: str, model_id: str) -> None:
    out_dir = Path(out_dir)
    segments_dir = out_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": str(image_path),
        "caption": caption,
        "model_id": model_id,
        "ts": int(time.time()),
    }
    caption_path = segments_dir / "caption.json"
    with caption_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
