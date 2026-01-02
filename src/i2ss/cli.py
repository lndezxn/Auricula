"""Typer CLI that scaffolds segmentation and soundscape artifacts."""
from __future__ import annotations

import cv2
import json
import time
from pathlib import Path
from typing import Any, Literal

import typer
import torch

from .captioning import caption_image, save_caption
from .config import Config
from .prompting import (
    build_audio_prompts,
    build_mix_meta,
    infer_scene_tags,
    save_mix_meta,
    save_prompts,
)
from .audio.audioldm2 import AudioLDM2Generator
from .audio.mix import mix_soundscape
from .utils import audio_io, io as io_utils
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



def _segment_and_save(
    image: Path,
    queries: str,
    conf: Config,
    device: str,
    sam_device: str,
) -> tuple[list[dict[str, object]], list[str]]:
    normalized_queries = parse_text_queries(queries)
    if not normalized_queries:
        raise typer.BadParameter("queries cannot be empty")

    image_bgr = cv2.imread(str(image))
    if image_bgr is None:
        raise typer.BadParameter(f"Unable to load image at {image}")

    dino_model = sam_predictor = None
    try:
        dino_model, sam_predictor = load_models(device=device, sam_device=sam_device)
        objects = run_grounded_segment(image_bgr, normalized_queries, dino_model, sam_predictor)
        save_segmentation(conf.segments_dir, image, objects)
        return objects, normalized_queries
    finally:
        if sam_predictor is not None:
            del sam_predictor
        if dino_model is not None:
            del dino_model
        torch.cuda.empty_cache()


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


def _read_json_object(source: Path) -> dict[str, Any]:
    if not source.exists():
        raise typer.BadParameter(f"Missing JSON file: {source}")
    with source.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"Expected object payload in {source}")
    return payload


def _load_prompts(prompts_json: Path) -> dict[str, Any]:
    return _read_json_object(prompts_json)


def _render_tracks(conf: Config, prompts: dict[str, Any]) -> None:
    io_utils.ensure_dir(conf.tracks_dir)
    sample_rate = audio_io.TARGET_SAMPLE_RATE
    seconds = int(prompts.get("seconds", conf.seconds))
    seed = int(prompts.get("seed", conf.seed))
    generator = AudioLDM2Generator(device="cuda" if torch.cuda.is_available() else "cpu")

    background_cfg = prompts.get("background", {})
    background_negative = background_cfg.get("negative_prompt")
    background_wave, background_sr = generator.generate(
        background_cfg.get("prompt", "Background ambience"),
        seconds,
        seed,
        num_inference_steps=50,
        guidance_scale=3.5,
        negative_prompt=background_negative,
    )
    background_wave, _ = audio_io.prepare_waveform(background_wave, background_sr, seconds, target_sr=sample_rate)
    audio_io.write_audio(conf.tracks_dir / "background.wav", background_wave, sample_rate)

    for idx, obj in enumerate(prompts.get("objects", [])):
        track_path = audio_io.build_object_path(conf.tracks_dir, obj.get("id", idx), obj.get("label", "object"))
        duration = int(obj.get("seconds", seconds))
        obj_seed = seed + int(obj.get("id", idx)) + 1
        obj_wave, obj_sr = generator.generate(
            obj.get("prompt", obj.get("label", "object")),
            duration,
            obj_seed,
            num_inference_steps=50,
            guidance_scale=3.5,
            negative_prompt=obj.get("negative_prompt"),
        )
        obj_wave, _ = audio_io.prepare_waveform(obj_wave, obj_sr, duration, target_sr=sample_rate)
        audio_io.write_audio(track_path, obj_wave, sample_rate)


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



@app.command()
def segment(
    image: Path = typer.Option(..., exists=True, file_okay=True),
    queries: str = typer.Option(..., help="Dot-separated list of objects, e.g. car.person.dog"),
    out: Path | None = typer.Option(None, help="Optional override for the output root directory"),
    device: Literal["cpu", "cuda"] = typer.Option(
        "cpu",
        help="Device for GroundingDINO (cpu or cuda)",
    ),
    sam_device: Literal["cpu", "cuda"] = typer.Option(
        "cpu",
        help="Device for the SAM predictor (cpu or cuda); on 8GB 4060 Ti prefer cpu",
    ),
) -> None:
    conf = _prepare_config(out)
    objects, normalized_queries = _segment_and_save(image, queries, conf, device, sam_device)
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
    meta_json: Path = typer.Option(..., exists=True, file_okay=True),
    out: Path | None = typer.Option(None, help="Optional output root override"),
    steps: int = typer.Option(50, help="Diffusion steps used per track"),
    guidance_scale: float = typer.Option(3.5, help="Guidance scale used by AudioLDM2"),
) -> None:
    if steps <= 0:
        raise typer.BadParameter("steps must be greater than zero")
    if guidance_scale <= 0:
        raise typer.BadParameter("guidance_scale must be positive")
    conf = _prepare_config(out)
    prompts = _load_prompts(prompts_json)
    meta = _read_json_object(meta_json)
    background_prompt = prompts.get("background")
    if not isinstance(background_prompt, dict) or not background_prompt.get("prompt"):
        raise typer.BadParameter("prompts JSON requires background.prompt")
    objects_prompts = prompts.get("objects")
    if not isinstance(objects_prompts, list) or not objects_prompts:
        raise typer.BadParameter("prompts JSON requires objects list")
    meta_background = meta.get("background")
    if not isinstance(meta_background, dict) or "path" not in meta_background:
        raise typer.BadParameter("meta JSON requires background.path")
    generator = AudioLDM2Generator()
    background_seconds = int(prompts.get("seconds", conf.seconds))
    background_seed = int(prompts.get("seed", conf.seed))
    start = time.perf_counter()
    background_audio, background_sr = generator.generate(
        background_prompt["prompt"],
        background_seconds,
        background_seed,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        negative_prompt=background_prompt.get("negative_prompt"),
    )
    background_wave, sample_rate = audio_io.prepare_waveform(
        background_audio, background_sr, background_seconds
    )
    background_path = conf.tracks_dir / "background.wav"
    audio_io.write_audio(background_path, background_wave, sample_rate)
    background_last = audio_io.last_non_silent_time(background_wave, sample_rate)
    typer.echo(
        f"Background duration {len(background_wave) / sample_rate:.2f}s @ {sample_rate}Hz (last non-silent {background_last:.2f}s)"
    )
    meta_background["path"] = str(background_path)
    meta["sample_rate"] = sample_rate
    meta["seconds"] = background_seconds
    meta_objects: list[dict[str, Any]] = meta.setdefault("objects", [])
    meta_by_id = {
        obj.get("id"): obj
        for obj in meta_objects
        if isinstance(obj, dict) and obj.get("id") is not None
    }
    object_paths: list[Path] = []
    for idx, obj_prompt in enumerate(objects_prompts):
        if not isinstance(obj_prompt, dict) or not obj_prompt.get("prompt"):
            raise typer.BadParameter("each object prompt must contain id, label, seconds, prompt")
        obj_id = obj_prompt.get("id", idx)
        obj_label = obj_prompt.get("label", "object")
        obj_seconds = int(obj_prompt.get("seconds", background_seconds))
        if isinstance(obj_id, int):
            obj_seed = background_seed + obj_id + 1
        else:
            obj_seed = background_seed + idx + 1
        obj_negative = obj_prompt.get("negative_prompt")
        waveform, sr = generator.generate(
            obj_prompt["prompt"],
            obj_seconds,
            obj_seed,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt=obj_negative,
        )
        prepared_wave, sr = audio_io.prepare_waveform(waveform, sr, obj_seconds)
        obj_path = audio_io.build_object_path(conf.tracks_dir, obj_id, obj_label)
        audio_io.write_audio(obj_path, prepared_wave, sr)
        object_paths.append(obj_path)
        object_last = audio_io.last_non_silent_time(prepared_wave, sr)
        typer.echo(
            f"Object {obj_label} duration {len(prepared_wave) / sr:.2f}s @ {sr}Hz (last non-silent {object_last:.2f}s) -> {obj_path}"
        )
        meta_entry = meta_by_id.get(obj_id)
        if meta_entry is None:
            if idx < len(meta_objects):
                meta_entry = meta_objects[idx]
            else:
                meta_entry = {}
                meta_objects.append(meta_entry)
        meta_entry.setdefault("id", obj_id)
        meta_entry.setdefault("label", obj_label)
        meta_entry["path"] = str(obj_path)
    target_meta_path = conf.tracks_dir / "meta.json"
    io_utils.write_json(target_meta_path, meta)
    elapsed = time.perf_counter() - start
    typer.echo(f"Background audio written to {background_path}")
    for obj_path in object_paths:
        typer.echo(f"Object audio written to {obj_path}")
    typer.echo(f"Updated mix metadata -> {target_meta_path}")
    typer.echo(f"AudioLDM2 generation completed in {elapsed:.2f}s")


@app.command()
def mix(
    tracks_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    meta_json: Path = typer.Option(..., exists=True, file_okay=True),
    out: Path = typer.Option(..., help="Mix filename that lives inside the mix directory"),
) -> None:
    metadata = _load_meta(meta_json)
    mix_file = Path(out)
    mix_dir = mix_file.parent
    mix_dir.mkdir(parents=True, exist_ok=True)
    mix_soundscape(
        tracks_dir,
        meta_json,
        mix_file,
        target_sr=int(metadata.get("sample_rate", audio_io.TARGET_SAMPLE_RATE)),
        peak_db=float(metadata.get("mixing", {}).get("peak_db", -1.0)),
    )
    io_utils.write_json(mix_dir / "meta.json", metadata)
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
    device: Literal["cpu", "cuda"] = typer.Option(
        "cpu",
        help="Device for GroundingDINO (cpu or cuda)",
    ),
    sam_device: Literal["cpu", "cuda"] = typer.Option(
        "cpu",
        help="Device for the SAM predictor (cpu or cuda); on 8GB 4060 Ti prefer cpu",
    ),
) -> None:
    if seconds <= 0:
        raise typer.BadParameter("seconds must be greater than zero")
    conf = _prepare_config(out)
    objects, _ = _segment_and_save(image, queries, conf, device, sam_device)
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
