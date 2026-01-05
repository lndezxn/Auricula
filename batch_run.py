#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

def run_batch(
    batch_file: Path,
    vlm_device: str,
    device: str,
    sam_device: str,
    dry_run: bool = False
):
    if not batch_file.exists():
        print(f"Error: Batch file {batch_file} not found.")
        sys.exit(1)

    with open(batch_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    if not isinstance(tasks, list):
        print("Error: Batch file must contain a JSON list of tasks.")
        sys.exit(1)

    total = len(tasks)
    print(f"Found {total} tasks in {batch_file}")

    for i, task in enumerate(tasks):
        image = task.get("image")
        out = task.get("out")
        queries = task.get("queries")
        
        if not image or not out:
            print(f"Skipping task {i}: missing 'image' or 'out' field")
            continue
        
        if not queries:
            print(f"Skipping task {i}: missing 'queries' field (required for segmentation)")
            continue

        out_path = Path(out)
        if out_path.exists():
            print(f"\033[93mWARNING: Output directory {out} already exists. Skipping task {i}.\033[0m")
            continue

        cmd = [
            sys.executable, "-m", "i2ss.cli", "run",
            "--image", str(image),
            "--out", str(out),
            "--queries", str(queries),
            "--vlm-device", vlm_device,
            "--device", device,
            "--sam-device", sam_device,
        ]
        
        # Optional overrides from task
        if "scene_hint" in task:
            cmd.extend(["--scene-hint", task["scene_hint"]])
        
        print(f"[{i+1}/{total}] Processing {image} -> {out}")
        if dry_run:
            print(f"  Command (Run): {' '.join(cmd)}")
        else:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  Error processing task {i} (Run): {e}")
                # Continue to next task instead of crashing
                continue

        # Mix step
        out_path = Path(out)
        mix_cmd = [
            sys.executable, "-m", "i2ss.cli", "mix",
            "--tracks-dir", str(out_path / "tracks"),
            "--meta-json", str(out_path / "tracks" / "meta.json"),
            "--out", str(out_path / "mix" / "mix.wav"),
        ]

        if dry_run:
            print(f"  Command (Mix): {' '.join(mix_cmd)}")
        else:
            try:
                subprocess.run(mix_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  Error processing task {i} (Mix): {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch run i2ss.cli for multiple images.")
    parser.add_argument("batch_file", type=Path, help="JSON file containing list of tasks (image, out, queries)")
    parser.add_argument("--vlm-device", default="cuda:0", help="Device for VLM (default: cuda:0)")
    parser.add_argument("--device", default="cpu", help="Device for GroundingDINO (default: cpu)")
    parser.add_argument("--sam-device", default="cpu", help="Device for SAM (default: cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    
    args = parser.parse_args()
    
    run_batch(
        args.batch_file,
        args.vlm_device,
        args.device,
        args.sam_device,
        args.dry_run
    )
