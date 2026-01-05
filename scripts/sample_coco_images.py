#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Set
import shutil

try:
    from pycocotools.coco import COCO  # type: ignore
except ImportError:
    print("pycocotools not installed. Please install with: pip install pycocotools", file=sys.stderr)
    sys.exit(1)

ALLOWED_CATS = {"dog", "car", "person"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a small COCO subset with specific category combos.")
    parser.add_argument("--coco_dir", default="data/coco2017", type=Path, help="Root COCO directory")
    parser.add_argument("--split", default="val2017", choices=["val2017", "train2017"], help="COCO split")
    parser.add_argument("--out_dir", default="data/images", type=Path, help="Output directory for sampled images")
    parser.add_argument("--n_total", default=40, type=int, help="Total target images (20-60 recommended)")
    parser.add_argument(
        "--must_include_sets",
        nargs="+",
        default=["dog,car", "person,car", "dog,person"],
        help="Category combos to guarantee (comma-separated strings)",
    )
    parser.add_argument("--min_instances_per_image", default=2, type=int, help="Minimum allowed instances per image (counting allowed categories)")
    parser.add_argument("--min_box_area", default=5000, type=float, help="Filter out instances smaller than this area")
    return parser.parse_args()


def load_coco(coco_dir: Path, split: str) -> COCO:
    ann_file = coco_dir / "annotations" / f"instances_{split}.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"Missing annotations file: {ann_file}")
    return COCO(str(ann_file))


def category_name_to_id(coco: COCO, names: Set[str]) -> Dict[str, int]:
    name_to_id: Dict[str, int] = {}
    for cat in coco.loadCats(coco.getCatIds()):
        name_lower = cat["name"].lower()
        if name_lower in names:
            name_to_id[name_lower] = int(cat["id"])
    missing = names - set(name_to_id.keys())
    if missing:
        raise ValueError(f"Missing categories in COCO annotations: {sorted(missing)}")
    return name_to_id


def collect_image_stats(coco: COCO, allowed_ids: Set[int], id_to_name: Dict[int, str], min_box_area: float) -> Dict[int, Dict[str, any]]:
    stats: Dict[int, Dict[str, any]] = {}
    for img_id in coco.getImgIds():
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=None))
        cats: List[str] = []
        count = 0
        for ann in anns:
            cid = ann.get("category_id")
            if cid not in allowed_ids:
                continue
            area = float(ann.get("area", 0))
            if area < min_box_area:
                continue
            cats.append(id_to_name.get(cid, ""))
            count += 1
        stats[img_id] = {
            "categories": cats,
            "count": count,
        }
    return stats


def image_meets_requirements(img_stats: Dict[str, any], required: Set[str], min_instances: int) -> bool:
    cats = img_stats["categories"]
    present = set(cats)
    if not required.issubset(present):
        return False
    return img_stats["count"] >= min_instances


def select_images(
    stats: Dict[int, Dict[str, any]],
    required_sets: List[Set[str]],
    target_total: int,
    min_instances: int,
) -> List[int]:
    selected: List[int] = []
    used: Set[int] = set()
    per_set_target = math.ceil(target_total / max(len(required_sets), 1))

    for req in required_sets:
        candidates = [img_id for img_id, s in stats.items() if image_meets_requirements(s, req, min_instances)]
        taken = 0
        for img_id in candidates:
            if img_id in used:
                continue
            selected.append(img_id)
            used.add(img_id)
            taken += 1
            if taken >= per_set_target:
                break

    if len(selected) < target_total:
        all_cands = [img_id for img_id, s in stats.items() if s["count"] >= min_instances]
        for img_id in all_cands:
            if img_id in used:
                continue
            selected.append(img_id)
            used.add(img_id)
            if len(selected) >= target_total:
                break
    return selected[:target_total]


def copy_and_manifest(
    coco: COCO,
    img_ids: List[int],
    stats: Dict[int, Dict[str, any]],
    coco_dir: Path,
    split: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_info = {img["id"]: img for img in coco.loadImgs(img_ids)}
    manifest = []
    for img_id in img_ids:
        info = images_info[img_id]
        file_name = info["file_name"]
        # Support layouts: coco_dir/images/split or coco_dir/split
        cand1 = coco_dir / "images" / split / file_name
        cand2 = coco_dir / split / file_name
        src = cand1 if cand1.exists() else cand2
        if not src.exists():
            raise FileNotFoundError(f"Missing image file: {cand1} or {cand2}")
        dst = out_dir / file_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        contains = sorted(set(stats.get(img_id, {}).get("categories", [])))
        manifest.append(
            {
                "image_id": img_id,
                "file_name": file_name,
                "path": str(dst),
                "contains": contains,
            }
        )
    manifest_path = out_dir.parent / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"Saved manifest with {len(manifest)} entries to {manifest_path}")


def main() -> None:
    global args
    args = parse_args()
    coco = load_coco(args.coco_dir, args.split)
    name_to_id = category_name_to_id(coco, ALLOWED_CATS)
    allowed_ids = set(name_to_id.values())

    required_sets: List[Set[str]] = []
    for raw in args.must_include_sets:
        tokens = {t.strip().lower() for t in raw.split(",") if t.strip()}
        tokens = {t for t in tokens if t in ALLOWED_CATS}
        if tokens:
            required_sets.append(tokens)
    if not required_sets:
        raise ValueError("No valid must_include_sets provided")

    id_to_name = {v: k for k, v in name_to_id.items()}
    stats = collect_image_stats(coco, allowed_ids, id_to_name, args.min_box_area)
    img_ids = select_images(
        stats,
        required_sets,
        target_total=args.n_total,
        min_instances=args.min_instances_per_image,
    )
    if len(img_ids) < args.n_total:
        print(f"Warning: only found {len(img_ids)} images meeting criteria (requested {args.n_total})")

    copy_and_manifest(coco, img_ids, stats, args.coco_dir, args.split, args.out_dir)


if __name__ == "__main__":
    main()
