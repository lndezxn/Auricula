from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from ..utils import audio_io

TARGET_SAMPLE_RATE = audio_io.TARGET_SAMPLE_RATE


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    data, sample_rate = sf.read(str(path), dtype="float32")
    return np.asarray(data, dtype=np.float32), sample_rate


def ensure_length(waveform: np.ndarray, target_len_samples: int) -> np.ndarray:
    target_len = max(0, int(target_len_samples))
    current_len = waveform.shape[0]
    if current_len == target_len:
        return waveform
    if waveform.ndim == 1:
        if current_len > target_len:
            return waveform[:target_len]
        padding = np.zeros(target_len - current_len, dtype=waveform.dtype)
        return np.concatenate((waveform, padding))
    channels = waveform.shape[1]
    if current_len > target_len:
        return waveform[:target_len, :]
    padding = np.zeros((target_len - current_len, channels), dtype=waveform.dtype)
    return np.concatenate((waveform, padding), axis=0)


def stereo_pan(mono_wav: np.ndarray, pan: float) -> np.ndarray:
    mono = np.asarray(mono_wav, dtype=np.float32)
    pan = max(-1.0, min(1.0, float(pan)))
    angle = (pan + 1.0) * (math.pi / 4.0)
    left = mono * math.cos(angle)
    right = mono * math.sin(angle)
    return np.stack((left, right), axis=-1)


def _resolve_track_path(tracks_dir: Path, track_path: str | Path) -> Path:
    candidate = Path(track_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return tracks_dir / candidate


def _ensure_mono(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 1:
        return waveform
    return waveform.mean(axis=1)


def _db_to_linear(db: float) -> float:
    return 10 ** (db / 20)


def _area_gain_offset(area: float, min_area: float, max_area: float) -> float:
    if max_area <= min_area:
        return 0.0
    ratio = min(1.0, max(0.0, (area - min_area) / (max_area - min_area)))
    return -2.0 + 4.0 * ratio


def _ensure_stereo(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 1:
        return stereo_pan(waveform, 0.0)
    if waveform.shape[1] == 2:
        return waveform
    if waveform.shape[1] == 1:
        return stereo_pan(waveform[:, 0], 0.0)
    return waveform[:, :2]


def mix_soundscape(
    tracks_dir: Path,
    meta_json: Path,
    out_wav_path: Path,
    target_sr: int = TARGET_SAMPLE_RATE,
    peak_db: float = -1.0,
) -> None:
    tracks_dir = Path(tracks_dir)
    out_wav_path = Path(out_wav_path)
    meta_json = Path(meta_json)
    with meta_json.open(encoding="utf-8") as fp:
        metadata = json.load(fp)
    if not isinstance(metadata, dict):
        raise ValueError("Meta JSON must describe a mapping of mix information")
    seconds = int(metadata.get("seconds", 10))
    target_len = max(1, seconds * target_sr)
    background_info = metadata.get("background")
    if not background_info or "path" not in background_info:
        raise ValueError("Meta JSON must contain background.path")
    background_path = _resolve_track_path(tracks_dir, background_info["path"])
    background_wave, background_sr = load_wav(background_path)
    background_wave, _ = audio_io.prepare_waveform(
        background_wave, background_sr, seconds, target_sr=target_sr
    )
    background_stereo = _ensure_stereo(background_wave)
    background_gain = float(background_info.get("gain_db", 0.0))
    background_stereo *= _db_to_linear(background_gain)
    background_stereo = ensure_length(background_stereo, target_len)
    stems_dir = out_wav_path.parent / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(stems_dir / "background.wav"), background_stereo, target_sr)
    objects = metadata.get("objects") or []
    area_values = [float(obj.get("area", 0)) for obj in objects]
    min_area = min(area_values, default=0.0)
    max_area = max(area_values, default=min_area)
    mix_buffer = np.array(background_stereo, copy=True)
    for idx, obj in enumerate(objects):
        track_path = obj.get("path")
        if not track_path:
            continue
        obj_wave, obj_sr = load_wav(_resolve_track_path(tracks_dir, track_path))
        obj_seconds = int(obj.get("seconds", seconds))
        obj_wave, _ = audio_io.prepare_waveform(obj_wave, obj_sr, obj_seconds, target_sr=target_sr)
        mono_wave = _ensure_mono(obj_wave)
        centroid_x = float(obj.get("centroid_x", 0.5))
        pan = max(-1.0, min(1.0, (centroid_x - 0.5) * 2.0))
        obj_stereo = stereo_pan(mono_wave, pan)
        obj_gain = float(obj.get("gain_db", 0.0))
        area = float(obj.get("area", 0.0))
        obj_stereo *= _db_to_linear(obj_gain + _area_gain_offset(area, min_area, max_area))
        obj_stereo = ensure_length(obj_stereo, target_len)
        obj_id = obj.get("id", idx)
        obj_label = obj.get("label", "object")
        stem_name = f"{obj_id}_{audio_io.sanitize_label(obj_label)}.wav"
        sf.write(str(stems_dir / stem_name), obj_stereo, target_sr)
        mix_buffer += obj_stereo
    peak_limit = 10 ** (peak_db / 20)
    maximum = float(np.max(np.abs(mix_buffer)))
    if maximum > 0 and maximum > peak_limit:
        mix_buffer *= peak_limit / maximum
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav_path), mix_buffer, target_sr)
