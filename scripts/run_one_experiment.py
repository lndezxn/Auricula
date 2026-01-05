#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_NAME = "default"
COMMON_PROMPTS_PATHS = [
    Path("tracks") / "prompts.json",
    Path("prompts.json"),
]
COMMON_MIX_PATHS = [
    Path("mix") / "mix.wav",
    Path("mix.wav"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a small batch experiment with i2ss CLI.")
    p.add_argument("--manifest", default="data/manifest.json", type=Path)
    p.add_argument("--out_root", default="runs/exp_single", type=Path)
    p.add_argument("--queries", default="dog.car.person")
    p.add_argument("--seconds", default=10, type=int)
    p.add_argument("--limit", default=20, type=int)
    p.add_argument("--device", default="cpu", help="GroundingDINO device")
    p.add_argument("--sam-device", default="cpu", help="SAM device")
    p.add_argument("--vlm-device", default="cuda:0", help="VLM device")
    return p.parse_args()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("manifest must be a list")
    return data


def run_case(
    img_path: Path,
    out_dir: Path,
    queries: str,
    seconds: int,
    device: str,
    sam_device: str,
    vlm_device: str,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "i2ss.cli",
        "run",
        "--image",
        str(img_path),
        "--queries",
        queries,
        "--out",
        str(out_dir),
        "--device",
        device,
        "--sam-device",
        sam_device,
        "--vlm-device",
        vlm_device,
    ]
    # pass seconds if supported; harmless if ignored
    cmd.extend(["--seconds", str(seconds)])
    log_path = out_dir / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fp:
        result = subprocess.run(cmd, stdout=log_fp, stderr=subprocess.STDOUT)
    return result.returncode


def find_first(path: Path, candidates: List[Path]) -> Optional[Path]:
    for rel in candidates:
        cand = path / rel
        if cand.exists():
            return cand
    return None


def read_prompts(prompts_path: Path) -> Dict[str, Any]:
    with prompts_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def detect_contamination(sound_prompt: str, other_labels: List[str]) -> Optional[bool]:
    if not isinstance(sound_prompt, str):
        return None
    text = sound_prompt.lower()
    for label in other_labels:
        if not label:
            continue
        base = label.lower()
        variants = {base, f"{base}s", f"{base}es"}
        for v in variants:
            if v in text:
                return True
    return False


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    results_rows: List[Dict[str, Any]] = []
    total_images = 0
    prompts_found = 0
    mix_found = 0
    total_objects = 0
    total_sound_ok = 0
    contaminated_objects = 0

    for idx, item in enumerate(manifest):
        if idx >= args.limit:
            break
        img_path = Path(item.get("path") or item.get("file_name") or "")
        if not img_path.exists():
            print(f"[warn] missing image for idx {idx}: {img_path}")
            continue
        total_images += 1
        case_dir = out_root / f"{idx:04d}"
        exit_code = run_case(
            img_path,
            case_dir,
            args.queries,
            args.seconds,
            args.device,
            args.sam_device,
            args.vlm_device,
        )
        prompts_path = find_first(case_dir, COMMON_PROMPTS_PATHS)
        mix_path = find_first(case_dir, COMMON_MIX_PATHS)
        prompts_json_found = prompts_path is not None
        mix_wav_found = mix_path is not None
        if prompts_json_found:
            prompts_found += 1
        if mix_wav_found:
            mix_found += 1
        objects: List[Dict[str, Any]] = []
        if prompts_path:
            try:
                payload = read_prompts(prompts_path)
                objects = payload.get("objects", []) or []
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] failed to read prompts for idx {idx}: {exc}")
        # build rows
        labels = [str(o.get("label", "object")) for o in objects if isinstance(o, dict)]
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            total_objects += 1
            label = str(obj.get("label", "object"))
            other = [l for l in labels if l != label]
            sound_prompt = obj.get("sound_prompt") or obj.get("prompt") or ""
            contaminated = detect_contamination(sound_prompt, other)
            if isinstance(sound_prompt, str) and sound_prompt:
                total_sound_ok += 1
                if contaminated:
                    contaminated_objects += 1
            results_rows.append(
                {
                    "config_name": CONFIG_NAME,
                    "idx": idx,
                    "image_path": str(img_path),
                    "case_out_dir": str(case_dir),
                    "label": label,
                    "other_labels": ";".join(other),
                    "sound_prompt": sound_prompt,
                    "contaminated": contaminated if contaminated is not None else "",
                    "prompts_json_found": prompts_json_found,
                    "mix_wav_found": mix_wav_found,
                    "run_exit_code": exit_code,
                }
            )
        if not objects:
            results_rows.append(
                {
                    "config_name": CONFIG_NAME,
                    "idx": idx,
                    "image_path": str(img_path),
                    "case_out_dir": str(case_dir),
                    "label": "",
                    "other_labels": "",
                    "sound_prompt": "",
                    "contaminated": "",
                    "prompts_json_found": prompts_json_found,
                    "mix_wav_found": mix_wav_found,
                    "run_exit_code": exit_code,
                }
            )

    # write CSV
    results_csv = out_root / "results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_name",
        "idx",
        "image_path",
        "case_out_dir",
        "label",
        "other_labels",
        "sound_prompt",
        "contaminated",
        "prompts_json_found",
        "mix_wav_found",
        "run_exit_code",
    ]
    with results_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    # summary
    summary_md = out_root / "summary.md"
    contam_rate = (contaminated_objects / total_sound_ok) if total_sound_ok else 0.0
    lines = []
    lines.append("# Experiment Summary\n")
    lines.append(f"- total_images: {total_images}")
    lines.append(f"- prompts_json_found: {prompts_found}")
    lines.append(f"- mix_wav_found: {mix_found}")
    lines.append(f"- total_objects: {total_objects}")
    lines.append(f"- sound_prompts_available: {total_sound_ok}")
    lines.append(f"- contaminated_objects: {contaminated_objects}")
    lines.append(f"- contam_rate: {contam_rate:.4f}\n")
    lines.append("| metric | value |\n|---|---|\n")
    metrics = {
        "total_images": total_images,
        "prompts_json_found": prompts_found,
        "mix_wav_found": mix_found,
        "total_objects": total_objects,
        "sound_prompts_available": total_sound_ok,
        "contaminated_objects": contaminated_objects,
        "contam_rate": f"{contam_rate:.4f}",
    }
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |\n")

    # pick 3 examples (prefer contaminated)
    contaminated_rows = [r for r in results_rows if r.get("contaminated") is True]
    sample_rows = contaminated_rows[:3] if contaminated_rows else random.sample(results_rows, k=min(3, len(results_rows))) if results_rows else []
    if sample_rows:
        lines.append("\n## Samples\n")
        for r in sample_rows:
            lines.append(f"- image_path: {r['image_path']}")
            lines.append(f"  - label: {r['label']}")
            lines.append(f"  - sound_prompt: {r['sound_prompt']}")
            lines.append(f"  - contaminated: {r['contaminated']}\n")

    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"results_csv: {results_csv}")
    print(f"summary_md:  {summary_md}")


if __name__ == "__main__":
    main()
