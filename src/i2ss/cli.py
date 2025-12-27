"""Typer CLI that scaffolds segmentation and soundscape artifacts."""
from __future__ import annotations

import cv2
import json
from pathlib import Path
from typing import Any

import typer

from .captioning import caption_image, save_caption
from .config import Config
from .prompting import (
    build_audio_prompts,
    build_mix_meta,
    infer_scene_tags,
    object_track_path,
    save_mix_meta,
    save_prompts,
)
from .utils import io as io_utils
from .vision import (
    load_models,
    parse_text_queries,
    save_segmentation,
    segment as run_grounded_segment,
)

app = typer.Typer(help="""i2ss CLI for creating segmentation, track, and mix artifacts.""")


def _prepare_config(out_dir: Path | None = None) -> Config:
    overrides = {}
    if out_dir is not None:
        overrides["out_dir"] = out_dir
    config = Config.load(**overrides)
    config.ensure_dirs()
    return config



def _segment_and_save(image: Path, queries: str, conf: Config) -> tuple[list[dict[str, object]], list[str]]:
    normalized_queries = parse_text_queries(queries)
    if not normalized_queries:
        raise typer.BadParameter("queries cannot be empty")

    image_bgr = cv2.imread(str(image))
    if image_bgr is None:
        raise typer.BadParameter(f"Unable to load image at {image}")

    dino_model, sam_predictor = load_models()
    objects = run_grounded_segment(image_bgr, normalized_queries, dino_model, sam_predictor)
    save_segmentation(conf.segments_dir, image, objects)
    return objects, normalized_queries


def _generate_caption(image: Path, conf: Config, model_id: str) -> str:
    caption_text = caption_image(str(image), model_id=model_id)
    save_caption(conf.out_dir, image, caption_text, model_id)
    return caption_text


def _serialize_segmentation_objects(objects: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for idx, obj in enumerate(objects):
        normalized.append(
            {
                "id": obj.get("id", idx),
                "label": obj.get("label", "object"),
                "centroid_x": float(obj.get("centroid_x", 0.5)),
                "centroid_y": float(obj.get("centroid_y", 0.5)),
                "area": int(max(0, obj.get("area", 0))),
            }
        )
    return normalized


def _write_segments_context(
    conf: Config,
    image: Path,
    objects: list[dict[str, object]],
    caption_text: str,
    scene_hint: str | None,
) -> Path:
    entries = _serialize_segmentation_objects(objects)
    labels = [entry.get("label", "object") for entry in entries]
    prompt_caption = " ".join(filter(None, [caption_text, scene_hint or ""]))
    if scene_hint:
        labels.append(scene_hint)
    scene_tags = infer_scene_tags(prompt_caption, labels)
    payload = {
        "image_path": str(image),
        "caption": caption_text,
        "scene_tags": scene_tags,
        "objects": entries,
    }
    meta_path = conf.segments_dir / "meta.json"
    io_utils.write_json(meta_path, payload)
    return meta_path


def _load_prompts(prompts_json: Path) -> dict[str, Any]:
    if not prompts_json.exists():
        raise typer.BadParameter(f"Missing prompts file: {prompts_json}")
    with prompts_json.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"Expected object payload in {prompts_json}")
    return payload


def _render_tracks(conf: Config, prompts: dict[str, Any]) -> None:
    io_utils.ensure_dir(conf.tracks_dir)
    seconds = int(prompts.get("seconds", conf.seconds))
    seed = int(prompts.get("seed", conf.seed))
    background_prompt = prompts.get("background", {}).get("prompt", "Background ambience")
    io_utils.write_text(
        conf.tracks_dir / "background.wav",
        f"Background track | {seconds}s @ seed {seed} | prompt: {background_prompt}",
    )
    for obj in prompts.get("objects", []):
        track_path = object_track_path(conf.tracks_dir, obj)
        duration = int(obj.get("seconds", seconds))
        prompt_text = obj.get("prompt", obj.get("label", "object"))
        io_utils.write_text(
            track_path,
            f"Object track for {obj.get('label', 'object')} | {duration}s @ seed {seed} | prompt: {prompt_text}",
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
    objects, normalized_queries = _segment_and_save(image, queries, conf)
    typer.echo(
        f"Segmentation artifacts written to {conf.segments_dir} ({len(objects)} detections)"
    )


@app.command()
def caption(
    image: Path = typer.Option(..., exists=True, file_okay=True),
    out: Path | None = typer.Option(None, help="Output directory that holds the segments subfolder"),
    model_id: str = typer.Option(
        "Salesforce/blip-image-captioning-large",
        help="Transformer caption model to use",
    ),
) -> None:
    conf = _prepare_config(out)
    _generate_caption(image, conf, model_id)
    typer.echo(f"Caption saved to {conf.segments_dir / 'caption.json'}")


@app.command()
def generate(
    prompts_json: Path = typer.Option(..., exists=True, file_okay=True),
    out: Path | None = typer.Option(None, help="Optional output root override"),
    seconds: int | None = typer.Option(None, help="Duration override in seconds"),
    seed: int | None = typer.Option(None, help="Random seed override"),
) -> None:
    conf = _prepare_config(out)
    prompts = _load_prompts(prompts_json)
    if seconds is not None:
        prompts["seconds"] = max(1, seconds)
    if seed is not None:
        prompts["seed"] = seed
    save_prompts(conf.tracks_dir, prompts)
    mix_meta = build_mix_meta(prompts, conf.tracks_dir)
    save_mix_meta(conf.tracks_dir, mix_meta)
    _render_tracks(conf, prompts)
    typer.echo(
        f"Generated artifacts -> prompts: {conf.tracks_dir / 'prompts.json'}, meta: {conf.tracks_dir / 'meta.json'}"
    )


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
    caption_model_id: str = typer.Option(
        "Salesforce/blip-image-captioning-large",
        help="Caption model used to describe the scene",
    ),
    scene_hint: str | None = typer.Option(
        None,
        help="Optional hint that gets appended to the caption before prompt building",
    ),
    seconds: int = typer.Option(10, help="Duration used in prompt templates"),
) -> None:
    if seconds <= 0:
        raise typer.BadParameter("seconds must be greater than zero")
    conf = _prepare_config(out)
    objects, _ = _segment_and_save(image, queries, conf)
    caption_text = _generate_caption(image, conf, caption_model_id)
    segments_meta = _write_segments_context(conf, image, objects, caption_text, scene_hint)
    prompts = build_audio_prompts(
        segments_context_json_path=segments_meta,
        seconds=seconds,
        seed=conf.seed,
        scene_hint=scene_hint,
    )
    save_prompts(conf.tracks_dir, prompts)
    mix_meta = build_mix_meta(prompts, conf.tracks_dir)
    save_mix_meta(conf.tracks_dir, mix_meta)
    conf.seconds = seconds
    _render_tracks(conf, prompts)
    typer.echo(
        f"Run completed: {len(objects)} detections -> {conf.segments_dir}, tracks -> {conf.tracks_dir}, prompts -> {conf.tracks_dir / 'prompts.json'}, meta -> {conf.tracks_dir / 'meta.json'}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
