from __future__ import annotations

import numpy as np

from i2ss.audio.audioldm2 import AudioLDM2Generator


def metrics(x: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    if sr == 16000:
        x = x[::2]
        sr = 8000
    x = x.astype(np.float32)
    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))

    n = x.shape[0]
    win = np.hanning(n)
    X = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    band = (freqs >= 500.0) & (freqs <= 4000.0)
    mag = np.abs(X[band])
    f = freqs[band]
    idx = int(np.argmax(mag))
    best_f = float(f[idx])
    norm = float(mag[idx] / (np.median(mag) + 1e-9))
    return rms, peak, best_f, norm


def main() -> None:
    prompt = "Footsteps on pavement, outdoor city ambience, realistic field recording"
    neg = "low quality, distortion, ringing, beeping, high-pitched tone"

    gen = AudioLDM2Generator(device="cuda", torch_dtype="float16")
    gen.pipe.set_progress_bar_config(disable=True)

    for guidance in [1.0, 2.0, 3.5, 5.0, 7.0]:
        wav, sr = gen.generate(
            prompt,
            seconds=3,
            seed=0,
            num_inference_steps=30,
            guidance_scale=guidance,
            negative_prompt=neg,
        )
        wav = np.asarray(wav).reshape(-1)
        rms, peak, best_hz, norm_peak = metrics(wav, sr)
        print(
            f"guidance={guidance:>3} rms={rms:.4f} peak={peak:.4f} bestHz={best_hz:.1f} normPeak={norm_peak:.1f}"
        )


if __name__ == "__main__":
    main()
