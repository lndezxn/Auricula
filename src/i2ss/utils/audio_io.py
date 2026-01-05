from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

TARGET_SAMPLE_RATE = 16000


def sanitize_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    cleaned = cleaned.strip("_")
    return cleaned or "object"


def _resample_waveform(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    if waveform.ndim == 1:
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    channels = []
    for channel in range(waveform.shape[1]):
        resampled = librosa.resample(
            waveform[:, channel], orig_sr=orig_sr, target_sr=target_sr
        )
        channels.append(resampled)
    return np.stack(channels, axis=1)


def _fix_length(waveform: np.ndarray, length: int) -> np.ndarray:
    current = waveform.shape[0]
    if current == length:
        return waveform
    if current > length:
        return waveform[:length]
    pad_width = length - current
    if waveform.ndim == 1:
        return np.pad(waveform, (0, pad_width), mode="constant")
    pad_shape = ((0, pad_width), (0, 0))
    return np.pad(waveform, pad_shape, mode="constant")


def prepare_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    seconds: int,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    normalized = waveform.astype(np.float32, copy=False)
    resampled = _resample_waveform(normalized, sample_rate, target_sr)
    length = max(1, int(seconds) * target_sr)
    enforced = _fix_length(resampled, length)
    return enforced, target_sr


def last_non_silent_time(waveform: np.ndarray, sample_rate: int, threshold: float = 1e-4) -> float:
    """Return the timestamp of the last sample whose amplitude exceeds <threshold>."""
    if waveform.size == 0:
        return 0.0
    non_silent = np.abs(waveform) > threshold
    indices = np.nonzero(non_silent)[0]
    if indices.size == 0:
        return 0.0
    return float(indices[-1]) / sample_rate


def write_audio(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sample_rate)


def build_object_path(tracks_dir: Path, obj_id: int | str, label: str) -> Path:
    sanitized = sanitize_label(label)
    return tracks_dir / "objects" / f"{obj_id}_{sanitized}.wav"


def tonality_ratio(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    fmin: float = 500.0,
    fmax: float = 4000.0,
) -> tuple[float, float]:
    """Return (dominant_frequency_hz, peak_to_median_ratio) in a mid/high band.

    This is a cheap heuristic to catch the "piercing periodic tone" failure mode.
    Higher ratios indicate a more tone-like (narrow-band) signal.
    """

    if waveform.size == 0 or sample_rate <= 0:
        return 0.0, 0.0
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if x.size < 128:
        return 0.0, 0.0

    x = x - float(np.mean(x))
    n = int(x.shape[0])
    win = np.hanning(n).astype(np.float32, copy=False)
    X = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return 0.0, 0.0
    mag = np.abs(X[band])
    peak_idx = int(np.argmax(mag))
    peak = float(mag[peak_idx])
    median = float(np.median(mag) + 1e-9)
    dominant_hz = float(freqs[band][peak_idx])
    return dominant_hz, float(peak / median)
