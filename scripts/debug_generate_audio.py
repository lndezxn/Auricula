from __future__ import annotations

from pathlib import Path

from i2ss.audio.audioldm2 import AudioLDM2Generator
from i2ss.utils import audio_io


def main() -> None:
    out_dir = Path("out_audio_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = "Footsteps on pavement, outdoor city ambience, realistic field recording"
    negative = "low quality, distortion, beeping, ringing, tone, hum, artifacts"

    settings = [
        ("fp16", dict(device="cuda", torch_dtype="float16")),
        ("fp32", dict(device="cuda", torch_dtype="float32")),
    ]

    for tag, kwargs in settings:
        gen = AudioLDM2Generator(**kwargs)
        wav, sr = gen.generate(
            prompt,
            seconds=3,
            seed=0,
            num_inference_steps=150,
            guidance_scale=6.0,
            negative_prompt=negative,
        )
        wav, sr = audio_io.prepare_waveform(
            wav, sr, seconds=3, target_sr=audio_io.TARGET_SAMPLE_RATE
        )
        path = out_dir / f"{tag}_footsteps.wav"
        audio_io.write_audio(path, wav, sr)
        print(f"wrote {path} sr={sr} samples={len(wav)}")


if __name__ == "__main__":
    main()
