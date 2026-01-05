from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from i2ss.audio.audioldm2 import AudioLDM2Generator


@dataclass
class EmbedStats:
    shape: tuple
    mean: float
    std: float
    l2: float


def _tensor_stats(x: torch.Tensor) -> EmbedStats:
    xf = x.detach().float().cpu().flatten()
    mean = float(xf.mean())
    std = float(xf.std())
    l2 = float(torch.linalg.vector_norm(xf))
    return EmbedStats(tuple(x.shape), mean, std, l2)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    # Prompts can produce different sequence lengths. Compare mean-pooled embeddings.
    if a.ndim >= 2:
        a = a.mean(dim=1)
    if b.ndim >= 2:
        b = b.mean(dim=1)
    af = a.detach().float().cpu().flatten()
    bf = b.detach().float().cpu().flatten()
    denom = float(torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf))
    if denom == 0.0:
        return float("nan")
    return float(torch.dot(af, bf) / denom)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "cvssp/audioldm2"
    gen = AudioLDM2Generator(model_id=model_id, device=device, torch_dtype="float16" if device == "cuda" else "float32")
    pipe = gen.pipe

    print("device", device, "dtype", dtype)
    print("model", getattr(pipe, "_name_or_path", None) or getattr(pipe, "name_or_path", None))

    # Component inventory
    comps = getattr(pipe, "components", None)
    if isinstance(comps, dict):
        print("\n== components ==")
        for k in sorted(comps.keys()):
            v = comps[k]
            print(f"{k}: {type(v)}")
    else:
        print("No pipe.components dict")

    lm = getattr(pipe, "language_model", None)
    print("\nlanguage_model:", type(lm))

    # Embedding check
    prompt_a = "Footsteps on pavement, outdoor city ambience, realistic field recording"
    prompt_b = "A roaring jet engine taking off, extremely loud, realistic field recording"
    negative = "low quality, distortion"

    if hasattr(pipe, "encode_prompt"):
        with torch.no_grad():
            out_a = pipe.encode_prompt(
                prompt=prompt_a,
                device=device,
                num_waveforms_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative,
            )
            out_b = pipe.encode_prompt(
                prompt=prompt_b,
                device=device,
                num_waveforms_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative,
            )

        # diffusers may return tuple; the first item is usually prompt_embeds
        if isinstance(out_a, tuple):
            emb_a = out_a[0]
            emb_b = out_b[0]
        else:
            emb_a = out_a
            emb_b = out_b

        print("\n== encode_prompt stats ==")
        print("A:", _tensor_stats(emb_a))
        print("B:", _tensor_stats(emb_b))
        print("cos_sim(A,B):", _cos_sim(emb_a, emb_b))

        pooled_a = emb_a.detach().float().mean(dim=1) if emb_a.ndim >= 2 else emb_a.detach().float()
        pooled_b = emb_b.detach().float().mean(dim=1) if emb_b.ndim >= 2 else emb_b.detach().float()
        diff = (pooled_a - pooled_b).abs()
        print("mean|A-B| (pooled):", float(diff.mean().cpu()))
        print("max|A-B| (pooled):", float(diff.max().cpu()))
    else:
        print("pipe has no encode_prompt; cannot check embeddings")


if __name__ == "__main__":
    main()
