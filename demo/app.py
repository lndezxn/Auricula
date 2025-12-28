from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from i2ss.audio.audioldm2 import AudioLDM2Generator
from i2ss.audio.mix import mix_soundscape
from i2ss.captioning import caption_image
from i2ss.demo.state import DemoState
from i2ss.prompting import build_audio_prompts, build_mix_meta, infer_scene_tags, save_mix_meta, save_prompts
from i2ss.utils import audio_io
from i2ss.vision.grounded_sam import (find_object_by_point, load_models, save_segmentation,
                                       segment)

DEFAULT_QUERIES = "car.person.dog.cat.bus.train.motorcycle.bird.wave.fire"
SELECTED_OVERLAY = "overlay_selected.png"


def _ensure_work_dirs(state: DemoState) -> None:
    for path in (state.work_dir, state.segments_dir, state.tracks_dir, state.mix_dir):
        path.mkdir(parents=True, exist_ok=True)


def _reset_track_artifacts(state: DemoState) -> None:
    for path in (state.tracks_dir, state.mix_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _format_object_options(objects: Iterable[dict[str, Any]]) -> list[str]:
    options: list[str] = []
    for obj in objects:
        label = obj.get("label", "object")
        score = float(obj.get("score", 0.0))
        options.append(f"{obj.get('id')}: {label} ({score:.2f})")
    return options


def _format_object_info(obj: dict[str, Any] | None) -> str:
    if obj is None:
        return "No object selected yet."
    return (
        f"**Label:** {obj.get('label', 'object')}  \n"
        f"**Score:** {obj.get('score', 0.0):.3f}  \n"
        f"**Area:** {obj.get('area', 0)}  \n"
        f"**Centroid X:** {obj.get('centroid_x', 0.5):.2f}"
    )


def _write_segments_context(state: DemoState, caption: str, scene_hint: str | None) -> Path:
    prompt_caption = " ".join(filter(None, [caption, scene_hint or ""]))
    labels = [obj.get("label", "object") for obj in state.objects]
    scene_tags = infer_scene_tags(prompt_caption, labels)
    context = {
        "image_path": str(state.segments_dir / "image.png"),
        "caption": caption,
        "scene_tags": scene_tags,
        "objects": [
            {
                "id": obj.get("id"),
                "label": obj.get("label"),
                "score": obj.get("score", 0.0),
                "area": obj.get("area", 0),
                "centroid_x": obj.get("centroid_x", 0.5),
                "centroid_y": obj.get("centroid_y", 0.5),
                "mask_path": obj.get("mask_path"),
            }
            for obj in state.objects
        ],
    }
    context_path = state.segments_dir / "meta.json"
    with context_path.open("w", encoding="utf-8") as fp:
        json.dump(context, fp, ensure_ascii=False, indent=2)
    state.segments_context_path = context_path
    return context_path


def _highlight_overlay(state: DemoState, selected_id: int | None) -> Path:
    base = state.segments_dir / "overlay.png"
    destination = state.segments_dir / SELECTED_OVERLAY
    if not base.exists():
        return base
    overlay = Image.open(base).convert("RGBA")
    mask = state.masks.get(selected_id) if selected_id is not None else None
    if mask is not None:
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255)).convert("L")
        highlight = Image.new("RGBA", overlay.size, (255, 0, 0, 96))
        overlay = Image.composite(highlight, overlay, mask_image)
    overlay.save(destination)
    state.selected_overlay_path = destination
    return destination


def _load_objects_with_masks(state: DemoState) -> list[dict[str, Any]]:
    objects_path = state.segments_dir / "objects.json"
    if not objects_path.exists():
        return []
    with objects_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    entries = payload.get("objects", [])
    state.objects.clear()
    state.masks.clear()
    for idx, entry in enumerate(entries):
        mask_path = state.segments_dir / entry.get("mask_path", "")
        mask = None
        if mask_path.exists():
            mask_img = Image.open(mask_path).convert("L")
            mask = np.asarray(mask_img) > 0
        ui_entry = {
            "id": idx,
            "label": entry.get("label", "object"),
            "score": float(entry.get("score", 0.0)),
            "area": int(entry.get("area", 0)),
            "centroid_x": float(entry.get("centroid_x", 0.5)),
            "centroid_y": float(entry.get("centroid_y", 0.5)),
            "mask_path": entry.get("mask_path"),
        }
        if mask is not None:
            ui_entry["mask"] = mask
            state.masks[idx] = mask
        state.objects.append(ui_entry)
    return state.objects


def _segment_handler(
    image_path: str | None,
    queries: str,
    scene_hint: str | None,
    device: str,
    state: DemoState,
) -> tuple[Path, dict[str, Any], str, None, None, DemoState]:
    if not image_path:
        raise ValueError("Please upload an image before segmenting.")
    queries = queries or DEFAULT_QUERIES
    _ensure_work_dirs(state)
    _reset_track_artifacts(state)
    dest_image = state.segments_dir / "image.png"
    shutil.copy(image_path, dest_image)
    caption = caption_image(str(dest_image))
    dino_model, sam_predictor = load_models(device=device)
    image_bgr = cv2.imread(str(dest_image))
    if image_bgr is None:
        raise ValueError("Unable to read the uploaded image file.")
    objects = segment(image_bgr, queries, dino_model, sam_predictor)
    if not objects:
        raise ValueError("GroundingDINO did not detect any of the requested queries.")
    save_segmentation(state.segments_dir, dest_image, objects)
    _load_objects_with_masks(state)
    state.caption = caption
    state.scene_hint = scene_hint
    _write_segments_context(state, caption, scene_hint)
    if state.objects:
        state.selected_id = state.objects[0].get("id")
    else:
        state.selected_id = None
    overlay_path = _highlight_overlay(state, state.selected_id)
    options = _format_object_options(state.objects)
    selected_label = options[0] if options else None
    selected_option_label = _option_label_from_obj(state.objects[0]) if state.objects else None
    selected_info = _format_object_info(state.objects[0] if state.objects else None)
    state.last_mix_path = None
    return (
        overlay_path,
        gr.update(choices=options, value=selected_option_label),
        selected_info,
        None,
        None,
        state,
    )


def _ensure_prompts(state: DemoState) -> dict[str, Any]:
    prompts_path = state.tracks_dir / "prompts.json"
    if prompts_path.exists():
        with prompts_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    if not state.segments_context_path:
        raise ValueError("Please run Segment before generating prompts.")
    prompts = build_audio_prompts(
        state.segments_context_path,
        seconds=state.seconds,
        seed=state.seed,
        scene_hint=state.scene_hint,
    )
    save_prompts(state.tracks_dir, prompts)
    return prompts


def _generate_full_soundscape(state: DemoState, steps: int = 100, guidance_scale: float = 3.5) -> tuple[str, DemoState]:
    if not state.objects:
        raise ValueError("No segmentation context found. Please run Segment first.")
    prompts = _ensure_prompts(state)
    generator = AudioLDM2Generator()
    seconds = int(prompts.get("seconds", state.seconds))
    base_seed = int(prompts.get("seed", state.seed))
    background_prompt = prompts["background"]
    background_audio, background_sr = generator.generate(
        background_prompt["prompt"],
        seconds,
        base_seed,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        negative_prompt=background_prompt.get("negative_prompt"),
    )
    background_wave, sample_rate = audio_io.prepare_waveform(
        background_audio, background_sr, seconds, target_sr=audio_io.TARGET_SAMPLE_RATE
    )
    background_path = state.tracks_dir / "background.wav"
    audio_io.write_audio(background_path, background_wave, sample_rate)
    objects = prompts.get("objects", [])
    for idx, obj in enumerate(objects):
        obj_id = obj.get("id", idx)
        obj_seed = base_seed + (obj_id if isinstance(obj_id, int) else idx) + 1
        obj_audio, obj_sr = generator.generate(
            obj["prompt"],
            int(obj.get("seconds", seconds)),
            obj_seed,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt=obj.get("negative_prompt"),
        )
        prepared_wave, sr = audio_io.prepare_waveform(
            obj_audio, obj_sr, int(obj.get("seconds", seconds)), target_sr=audio_io.TARGET_SAMPLE_RATE
        )
        obj_path = audio_io.build_object_path(state.tracks_dir, obj_id, obj.get("label", "object"))
        audio_io.write_audio(obj_path, prepared_wave, sr)
    meta = build_mix_meta(prompts, state.tracks_dir)
    meta_path = state.tracks_dir / "meta.json"
    save_mix_meta(state.tracks_dir, meta)
    mix_file = state.mix_dir / "mix.wav"
    mix_soundscape(
        state.tracks_dir,
        meta_path,
        mix_file,
        target_sr=int(meta.get("sample_rate", audio_io.TARGET_SAMPLE_RATE)),
        peak_db=float(meta.get("mixing", {}).get("peak_db", -1.0)),
    )
    state.last_mix_path = mix_file
    return str(mix_file), state


def _generate_selected_object(selected_value: str | None, state: DemoState) -> tuple[str, DemoState]:
    selected_id = _parse_dropdown_value(selected_value)
    if selected_id is None:
        raise ValueError("Please select an object first.")
    prompts = _ensure_prompts(state)
    objects = prompts.get("objects", [])
    obj_entry = next((obj for obj in objects if obj.get("id") == selected_id), None)
    if obj_entry is None:
        raise ValueError("Selected object prompt is missing.")
    label = obj_entry.get("label", "object")
    obj_path = audio_io.build_object_path(state.tracks_dir, selected_id, label)
    if obj_path.exists():
        return str(obj_path), state
    base_seed = int(prompts.get("seed", state.seed))
    if isinstance(selected_id, int):
        obj_seed = base_seed + selected_id + 1
    else:
        obj_seed = base_seed + objects.index(obj_entry) + 1
    generator = AudioLDM2Generator()
    obj_audio, obj_sr = generator.generate(
        obj_entry["prompt"],
        int(obj_entry.get("seconds", state.seconds)),
        obj_seed,
        negative_prompt=obj_entry.get("negative_prompt"),
    )
    prepared_wave, sr = audio_io.prepare_waveform(
        obj_audio, obj_sr, int(obj_entry.get("seconds", state.seconds)), target_sr=audio_io.TARGET_SAMPLE_RATE
    )
    audio_io.write_audio(obj_path, prepared_wave, sr)
    state.last_mix_path = None
    return str(obj_path), state


def _play_selected(selected_value: str | None, state: DemoState) -> str:
    selected_id = _parse_dropdown_value(selected_value)
    if selected_id is None:
        raise ValueError("Select an object before playing it.")
    obj = next((obj for obj in state.objects if obj.get("id") == selected_id), None)
    if obj is None:
        raise ValueError("Selected object does not exist anymore.")
    obj_path = audio_io.build_object_path(state.tracks_dir, obj.get("id"), obj.get("label", "object"))
    if not obj_path.exists():
        raise ValueError("This object has not been generated yet. Run Generate SELECTED first.")
    return str(obj_path)


def _remix(state: DemoState) -> tuple[str, DemoState]:
    meta_path = state.tracks_dir / "meta.json"
    background_path = state.tracks_dir / "background.wav"
    if not meta_path.exists() or not background_path.exists():
        raise ValueError("Please run Generate FULL before remixing.")
    with meta_path.open("r", encoding="utf-8") as fp:
        meta = json.load(fp)
    mix_file = state.mix_dir / "mix.wav"
    mix_soundscape(
        state.tracks_dir,
        meta_path,
        mix_file,
        target_sr=int(meta.get("sample_rate", audio_io.TARGET_SAMPLE_RATE)),
        peak_db=float(meta.get("mixing", {}).get("peak_db", -1.0)),
    )
    state.last_mix_path = mix_file
    return str(mix_file), state


def _handle_overlay_click(event: gr.SelectData, state: DemoState) -> tuple[Path | str, str | None, str, DemoState]:
    if event is None:
        raise ValueError("Click event missing coordinates")
    overlay_width, overlay_height = _get_overlay_size(state)
    x, y = _extract_click_coordinates(event, overlay_width, overlay_height)
    selected = find_object_by_point(state.objects, x, y)
    if selected is None:
        overlay_path = state.selected_overlay_path or (state.segments_dir / "overlay.png")
        info = "Click inside an object mask to select it."
        return overlay_path, gr.update(), info, state
    state.selected_id = selected.get("id")
    overlay_path = _highlight_overlay(state, state.selected_id)
    info = _format_object_info(selected)
    option_label = f"{selected.get('id')}: {selected.get('label', 'object')} ({selected.get('score', 0.0):.2f})"
    return overlay_path, option_label, info, state


def _parse_dropdown_value(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value.split(":", 1)[0])
    except ValueError:
        return None


def _get_overlay_size(state: DemoState) -> tuple[int, int]:
    overlay_path = state.selected_overlay_path or (state.segments_dir / SELECTED_OVERLAY)
    if not overlay_path.exists():
        overlay_path = state.segments_dir / "overlay.png"
    if not overlay_path.exists():
        raise ValueError("No overlay image available for selection.")
    with Image.open(overlay_path) as overlay:
        return overlay.width, overlay.height


def _extract_click_coordinates(event: gr.EventData, image_width: int, image_height: int) -> tuple[float, float]:
    data = getattr(event, "_data", {}) or {}
    event_index = getattr(event, "index", None)
    if isinstance(event_index, (tuple, list)) and len(event_index) >= 2:
        x = float(event_index[0])
        y = float(event_index[1])
        return _normalize_click_coordinates(x, y, image_width, image_height)
    for x_key, y_key in (
        ("x", "y"),
        ("X", "Y"),
        ("clientX", "clientY"),
        ("pageX", "pageY"),
    ):
        x = data.get(x_key)
        y = data.get(y_key)
        if isinstance(x, (float, int)) and isinstance(y, (float, int)):
            return _normalize_click_coordinates(float(x), float(y), image_width, image_height)
    value = data.get("value")
    if isinstance(value, dict):
        x = value.get("x") or value.get("X")
        y = value.get("y") or value.get("Y")
        if isinstance(x, (float, int)) and isinstance(y, (float, int)):
            return _normalize_click_coordinates(float(x), float(y), image_width, image_height)
    raise ValueError("Click event missing coordinates")


def _normalize_click_coordinates(x: float, y: float, image_width: int, image_height: int) -> tuple[float, float]:
    if 0 <= x <= 1 and 0 <= y <= 1:
        x *= image_width
        y *= image_height
    max_x = max(image_width - 1, 0)
    max_y = max(image_height - 1, 0)
    x = min(max(x, 0.0), max_x)
    y = min(max(y, 0.0), max_y)
    return x, y


def _option_label_from_obj(obj: dict[str, Any]) -> str:
    label = obj.get("label", "object")
    score = float(obj.get("score", 0.0))
    return f"{obj.get('id')}: {label} ({score:.2f})"


def _handle_dropdown_change(selected_value: str | None, state: DemoState) -> tuple[Path, str, DemoState]:
    selected_id = _parse_dropdown_value(selected_value)
    if selected_id is None:
        return state.selected_overlay_path or state.segments_dir / "overlay.png", _format_object_info(None), state
    state.selected_id = selected_id
    overlay_path = _highlight_overlay(state, selected_id)
    selected = next((obj for obj in state.objects if obj.get("id") == selected_id), None)
    info = _format_object_info(selected)
    return overlay_path, info, state


def main(device: str | None = None) -> None:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = DemoState()
    _ensure_work_dirs(state)

    with gr.Blocks(title="i2ss Gradio Demo") as demo:
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="Image", type="filepath")
                queries_in = gr.Textbox(label="Queries", value=DEFAULT_QUERIES)
                scene_hint_in = gr.Textbox(label="Scene hint", placeholder="Optional scene hint")
                device_in = gr.Textbox(label="Device", value=resolved_device)
                segment_btn = gr.Button("Segment")
                full_button = gr.Button("Generate FULL")
            with gr.Column():
                overlay_out = gr.Image(label="Overlay", type="filepath", interactive=False)
                info_box = gr.Markdown("No object selected yet.")
                obj_dropdown = gr.Dropdown(label="Objects", choices=[], interactive=True)
                generate_selected_btn = gr.Button("Generate SELECTED")
                play_selected_btn = gr.Button("Play SELECTED")
                mix_button = gr.Button("Mix NOW")
                selected_audio_out = gr.Audio(label="Selected stem", type="filepath")
                mix_audio_out = gr.Audio(label="Stereo mix", type="filepath")
        state_store = gr.State(state)

        segment_btn.click(
            fn=_segment_handler,
            inputs=[image_in, queries_in, scene_hint_in, device_in, state_store],
            outputs=[overlay_out, obj_dropdown, info_box, selected_audio_out, mix_audio_out, state_store],
        )
        obj_dropdown.change(
            fn=_handle_dropdown_change,
            inputs=[obj_dropdown, state_store],
            outputs=[overlay_out, info_box, state_store],
        )
        overlay_out.select(
            fn=_handle_overlay_click,
            inputs=[state_store],
            outputs=[overlay_out, obj_dropdown, info_box, state_store],
        )
        generate_selected_btn.click(
            fn=_generate_selected_object,
            inputs=[obj_dropdown, state_store],
            outputs=[selected_audio_out, state_store],
        )
        play_selected_btn.click(
            fn=_play_selected,
            inputs=[obj_dropdown, state_store],
            outputs=[selected_audio_out],
        )
        full_button.click(
            fn=_generate_full_soundscape,
            inputs=[state_store],
            outputs=[mix_audio_out, state_store],
        )
        mix_button.click(
            fn=_remix,
            inputs=[state_store],
            outputs=[mix_audio_out, state_store],
        )

    demo.launch(share=False)


if __name__ == "__main__":
    main()
