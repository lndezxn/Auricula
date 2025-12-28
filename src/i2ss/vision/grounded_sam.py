from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np
import torch
from torchvision.ops import nms

DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
REPO_ROOT = Path(__file__).resolve().parents[3]
THIRD_PARTY = REPO_ROOT / "third_party"
GROUNDINGDINO_ROOT = THIRD_PARTY / "GroundingDINO"
SAM_ROOT = THIRD_PARTY / "segment-anything"
_COLOR_PALETTE = (
    (65, 105, 225),
    (0, 191, 255),
    (50, 205, 50),
    (255, 165, 0),
    (255, 99, 71),
    (148, 0, 211),
    (245, 222, 179),
    (255, 215, 0),
)


def _ensure_sys_path(path: Path) -> None:
    normalized = str(path)
    if normalized and normalized not in sys.path:
        sys.path.insert(0, normalized)


def _resolve_device(device: str | None) -> str:
    if device is None:
        return DEFAULT_DEVICE
    return device


def parse_text_queries(raw_queries: str | Sequence[str]) -> list[str]:
    if isinstance(raw_queries, str):
        values = [raw_queries]
    else:
        values = list(raw_queries)

    parsed: list[str] = []
    for value in values:
        if value is None:
            continue
        for chunk in str(value).split("."):
            token = chunk.strip()
            if token:
                parsed.append(token)
    return parsed


def load_models(
    dino_repo_dir: Path | str | None = None,
    sam_ckpt: Path | str | None = None,
    device: str | None = None,
) -> tuple[Any, Any]:
    dino_repo_dir = Path(dino_repo_dir) if dino_repo_dir is not None else GROUNDINGDINO_ROOT
    sam_ckpt = Path(sam_ckpt) if sam_ckpt is not None else REPO_ROOT / "checkpoints" / "sam_vit_h_4b8939.pth"

    _ensure_sys_path(dino_repo_dir)
    _ensure_sys_path(SAM_ROOT)

    from groundingdino.util.inference import Model as GroundingDINOModel
    from segment_anything import SamPredictor, sam_model_registry

    config_path = dino_repo_dir / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    checkpoint_path = REPO_ROOT / "checkpoints" / "groundingdino_swint_ogc.pth"
    resolved_device = _resolve_device(device)

    dino_model = GroundingDINOModel(
        model_config_path=str(config_path),
        model_checkpoint_path=str(checkpoint_path),
        device=resolved_device,
    )

    sam_constructor = sam_model_registry.get("vit_h")
    if sam_constructor is None:
        raise RuntimeError("sam_model_registry is missing vit_h checkpoint constructor")
    sam_model = sam_constructor(checkpoint=str(sam_ckpt)).to(resolved_device)
    sam_predictor = SamPredictor(sam_model)

    return dino_model, sam_predictor


def segment(
    image_bgr: np.ndarray,
    text_queries: Sequence[str],
    dino_model: Any,
    sam_predictor: Any,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    nms_threshold: float = 0.5,
    max_detections: int | None = 64,
) -> list[dict[str, Any]]:
    queries = parse_text_queries(text_queries)
    if not queries:
        return []

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be an HxWx3 array")

    detections = dino_model.predict_with_classes(
        image=image_bgr,
        classes=queries,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    boxes_np = np.asarray(detections.xyxy, dtype=np.float32)
    if boxes_np.size == 0:
        return []

    scores_np = np.asarray(detections.confidence, dtype=np.float32)
    class_ids: list[int | None]
    raw_class_ids = getattr(detections, "class_id", None)
    if raw_class_ids is None:
        class_ids = [None] * len(boxes_np)
    else:
        class_ids = list(raw_class_ids)

    boxes_tensor = torch.from_numpy(boxes_np)
    scores_tensor = torch.from_numpy(scores_np)
    keep = nms(boxes_tensor, scores_tensor, nms_threshold)
    if keep.numel() == 0:
        return []

    if max_detections is not None:
        keep = keep[:max_detections]

    sam_predictor.set_image(image_bgr, image_format="BGR")

    objects: list[dict[str, Any]] = []
    for idx in keep.tolist():
        box = boxes_np[idx]
        mask_candidates, _, _ = sam_predictor.predict(box=box, multimask_output=False)
        mask = np.asarray(mask_candidates[0], dtype=bool)
        label = "object"
        class_id = class_ids[idx]
        if class_id is not None and 0 <= class_id < len(queries):
            label = queries[class_id]
        ys, xs = np.nonzero(mask)
        if xs.size and ys.size:
            centroid_x = float(xs.mean() / mask.shape[1])
            centroid_y = float(ys.mean() / mask.shape[0])
        else:
            centroid_x = 0.5
            centroid_y = 0.5
        objects.append(
            {
                "label": label,
                "score": float(scores_tensor[idx].item()),
                "box_xyxy": [float(value) for value in box.tolist()],
                "mask": mask,
                "area": int(mask.sum()),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
            }
        )

    return objects


def save_segmentation(
    out_dir: Path | str,
    image_path: Path | str,
    objects: Iterable[dict[str, Any]],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    image_path = Path(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image {image_path}")

    entries: list[dict[str, Any]] = []
    overlay = image.copy()

    for idx, obj in enumerate(objects):
        mask = np.asarray(obj.get("mask"), dtype=bool)
        label = obj.get("label", "object")
        safe_label = _sanitize_label(label)
        mask_name = f"{idx}_{safe_label}.png"
        mask_path = masks_dir / mask_name
        mask_uint8 = (mask.astype(np.uint8)) * 255
        cv2.imwrite(str(mask_path), mask_uint8)

        color = np.array(_COLOR_PALETTE[idx % len(_COLOR_PALETTE)], dtype=np.uint8)
        masked_indices = mask
        if masked_indices.any():
            overlay[masked_indices] = (
                (overlay[masked_indices].astype(np.float32) * 0.4)
                + (color * 0.6)
            ).clip(0, 255).astype(np.uint8)

        entries.append(
            {
                "label": label,
                "score": obj.get("score", 0.0),
                "box_xyxy": obj.get("box_xyxy", []),
                "area": obj.get("area", 0),
                "centroid_x": obj.get("centroid_x", 0.5),
                "centroid_y": obj.get("centroid_y", 0.5),
                "mask_path": str(Path("masks") / mask_name),
            }
        )

    overlay_path = out_dir / "overlay.png"
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    cv2.imwrite(str(overlay_path), blended)

    summary = {
        "objects": entries,
        "image": str(image_path),
    }
    objects_path = out_dir / "objects.json"
    with objects_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


def _sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label)
    return cleaned or "object"


CLICK_FUZZ_RADIUS = 6


def _mask_contains_point(mask: np.ndarray, row: int, col: int) -> bool:
    height, width = mask.shape[:2]
    if not (0 <= row < height and 0 <= col < width):
        return False
    return bool(mask[row, col])


def _mask_near_point(mask: np.ndarray, row: int, col: int, radius: int) -> bool:
    height, width = mask.shape[:2]
    center_row = min(max(row, 0), height - 1)
    center_col = min(max(col, 0), width - 1)
    top = max(center_row - radius, 0)
    bottom = min(center_row + radius + 1, height)
    left = max(center_col - radius, 0)
    right = min(center_col + radius + 1, width)
    return mask[top:bottom, left:right].any()


def find_object_by_point(objects: Iterable[dict[str, Any]], x: float, y: float) -> Optional[dict[str, Any]]:
    """Return the object whose mask contains (x, y), preferring smaller areas if multiple hit."""
    row = int(round(y))
    col = int(round(x))
    candidates: list[tuple[dict[str, Any], float, float]] = []
    for obj in objects:
        mask = obj.get("mask")
        if mask is None:
            continue
        if _mask_contains_point(mask, row, col):
            area = float(obj.get("area", mask.sum()))
            score = float(obj.get("score", 0.0))
            candidates.append((obj, area, score))
    if candidates:
        candidates.sort(key=lambda item: (item[1], -item[2]))
        return candidates[0][0]
    # Try expanding to nearby pixels in case the point is off by a few pixels.
    near_candidates: list[tuple[dict[str, Any], float, float]] = []
    for obj in objects:
        mask = obj.get("mask")
        if mask is None:
            continue
        if _mask_near_point(mask, row, col, radius=CLICK_FUZZ_RADIUS):
            area = float(obj.get("area", mask.sum()))
            score = float(obj.get("score", 0.0))
            near_candidates.append((obj, area, score))
    if not near_candidates:
        return None
    near_candidates.sort(key=lambda item: (item[1], -item[2]))
    return near_candidates[0][0]
