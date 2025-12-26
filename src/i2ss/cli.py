"""Typer CLI that scaffolds segmentation and soundscape artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import typer

from .config import Config
from .utils import io as io_utils
from .utils import viz

app = typer.Typer(help="""i2ss CLI for creating segmentation, track, and mix artifacts.""")


def _prepare_config(out_dir: Path | None = None) -> Config:
    overrides = {}
    if out_dir is not None:
        overrides["out_dir"] = out_dir
    config = Config.load(**overrides)
    config.ensure_dirs()
    return config


def _queries_list(value: str) -> list[str]:
    return [segment.strip() for segment in value.split(".") if segment.strip()]


def _slugify(value: str) -> str:
    candidate = "".join(ch if ch.isalnum() else "_" for ch in value.strip())
    return candidate or "object"


def _build_segment(conf: Config, image: Path, queries: str) -> list[str]:
    query_list = _queries_list(queries)
    io_utils.write_text(
        conf.segments_dir / "mask.png",
        f"Mask placeholder for {image} covering {len(query_list)} objects",
    )
    viz.create_overlay(conf.segments_dir / "overlay.png", f"Overlay for {image} ({queries})")
    io_utils.write_json(
        conf.segments_dir / "objects.json",
        {"image": str(image), **viz.describe_queries(query_list)},
    )
    return query_list


def _load_prompts(prompts_json: Path) -> list[str]:
    if prompts_json.exists():
        with prompts_json.open(encoding="utf-8") as fp:
            payload = json.load(fp)
        prompts = payload.get("prompts")
        if isinstance(prompts, Sequence):
            return [str(item).strip() for item in prompts if str(item).strip()]
    return []


def _render_tracks(conf: Config, prompts: Sequence[str], seconds: int, seed: int) -> None:
    safe_prompts = [prompt or "object" for prompt in prompts]
    if not safe_prompts:
        safe_prompts = ["object"]
    for prompt in safe_prompts:
        slug = _slugify(prompt)
        io_utils.write_text(
            conf.tracks_dir / f"{slug}.wav",
            f"Track for {prompt} | {seconds}s @ seed {seed}",
        )
    io_utils.write_text(
        conf.tracks_dir / "background.wav",
        f"Background track | {seconds}s @ seed {seed}",
    )
    io_utils.write_json(
        conf.tracks_dir / "prompts.json",
        {"prompts": safe_prompts, "seconds": seconds, "seed": seed},
    )


def _load_meta(meta_json: Path) -> dict[str, object]:
    if meta_json.exists():
        try:
            with meta_json.open(encoding="utf-8") as fp:
                value = json.load(fp)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {"source": str(meta_json), "note": "fallback metadata"}


def _collect_stems(tracks_dir: Path, stems_dir: Path) -> None:
    for wav_file in sorted(tracks_dir.glob("*.wav")):
        io_utils.write_text(stems_dir / wav_file.name, f"Stem derived from {wav_file.name}")


def _write_mix(mix_file: Path, tracks_dir: Path) -> None:
    io_utils.write_text(mix_file, f"Mix generated from tracks in {tracks_dir}")


@app.command()
def segment(
    image: Path = typer.Option(..., exists=True, file_okay=True),
    queries: str = typer.Option(..., help="Dot-separated list of objects, e.g. car.person.dog"),
    out: Path | None = typer.Option(None, help="Optional override for the output root directory"),
) -> None:
    conf = _prepare_config(out)
    query_list = _build_segment(conf, image, queries)
    typer.echo(f"Segment artifacts written to {conf.segments_dir} ({len(query_list)} objects)")


@app.command()
def generate(
    prompts_json: Path = typer.Option(..., exists=True, file_okay=True),
    out: Path | None = typer.Option(None, help="Optional output root override"),
    seconds: int | None = typer.Option(None, help="Duration override in seconds"),
    seed: int | None = typer.Option(None, help="Random seed override"),
) -> None:
    conf = _prepare_config(out)
    prompts = _load_prompts(prompts_json)
    if not prompts:
        prompts = [prompts_json.stem or "object"]
    total_seconds = seconds if seconds is not None else conf.seconds
    total_seed = seed if seed is not None else conf.seed
    _render_tracks(conf, prompts, total_seconds, total_seed)
    typer.echo(f"Track artifacts written to {conf.tracks_dir}")


@app.command()
def mix(
    tracks_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    meta_json: Path = typer.Option(..., exists=True, file_okay=True),
    out: Path = typer.Option(..., help="Mix filename that lives inside the mix directory"),
) -> None:
    conf = _prepare_config(None)
    metadata = _load_meta(meta_json)
    mix_dir = conf.mix_dir
    stems_dir = io_utils.ensure_dir(mix_dir / "stems")
    mix_file = mix_dir / out.name
    _write_mix(mix_file, tracks_dir)
    io_utils.write_json(mix_dir / "meta.json", metadata)
    _collect_stems(tracks_dir, stems_dir)
    typer.echo(f"Mix written to {mix_file}")


@app.command()
def run(
    image: Path = typer.Option(..., exists=True, file_okay=True),
    queries: str = typer.Option(..., help="Dot-separated objects string"),
    out: Path | None = typer.Option(None, help="Optional output root override"),
) -> None:
    conf = _prepare_config(out)
    query_list = _build_segment(conf, image, queries)
    _render_tracks(conf, query_list or ["object"], conf.seconds, conf.seed)
    typer.echo(
        f"Run completed: segments -> {conf.segments_dir}, tracks -> {conf.tracks_dir}",
    )
