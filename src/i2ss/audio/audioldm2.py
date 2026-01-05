from __future__ import annotations

import numpy as np
import torch
from diffusers import AudioLDM2Pipeline
from types import MethodType
from transformers.generation.utils import GenerationMixin
from transformers import AutoModelForCausalLM
from typing import Iterable, Tuple


TARGET_SAMPLE_RATE = 16000

class AudioLDM2Generator:
    def __init__(
        self,
        model_id: str = "cvssp/audioldm2",
        device: str = "cuda",
        torch_dtype: str = "float16",
    ) -> None:
        dtype_attr = getattr(torch, torch_dtype, torch.float16)
        self.device = torch.device(device)
        try:
            self.pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=dtype_attr)
        except TypeError:
            self.pipe = AudioLDM2Pipeline.from_pretrained(model_id, dtype=dtype_attr)
        self.pipe.to(self.device)
        if hasattr(self.pipe, "safety_checker"):
            self.pipe.safety_checker = None
        self._ensure_causal_language_model(model_id, dtype_attr)
        self._patch_language_model()

    def _ensure_causal_language_model(self, model_id: str, dtype_attr: torch.dtype) -> None:
        """Ensure the pipeline's language_model supports generation.

        With some dependency combinations, diffusers may load a plain GPT2Model, but
        AudioLDM2's prompt encoding calls generation-only helpers. Replacing it with
        a causal LM fixes encode_prompt and improves prompt conditioning stability.
        """

        language_model = getattr(self.pipe, "language_model", None)
        if language_model is None:
            return

        if hasattr(language_model, "generate") and hasattr(language_model, "_get_initial_cache_position"):
            return

        try:
            causal = AutoModelForCausalLM.from_pretrained(
                model_id,
                subfolder="language_model",
                torch_dtype=dtype_attr,
            )
            causal.to(self.device, dtype=dtype_attr)
            self.pipe.language_model = causal
            components = getattr(self.pipe, "components", None)
            if isinstance(components, dict):
                components["language_model"] = causal
        except Exception:
            # Fall back to patching GenerationMixin methods below.
            return

    def _patch_language_model(self) -> None:
        language_model = getattr(self.pipe, "language_model", None)
        if language_model is None:
            return
        for method_name in (
            "_get_initial_cache_position",
            "_update_model_kwargs_for_generation",
        ):
            if not hasattr(language_model, method_name):
                method = getattr(GenerationMixin, method_name)
                setattr(language_model, method_name, MethodType(method, language_model))

    # The pipeline always produces 16 kHz audio, so skip probing the feature extractor.

    def _prepare_audio_array(self, output_audio: Iterable[np.ndarray]) -> np.ndarray:
        array = np.asarray(output_audio)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 2 and array.shape[0] <= 2:
            array = array.transpose(1, 0)
        return array.astype(np.float32)

    def generate(
        self,
        prompt: str,
        seconds: int,
        seed: int,
        num_inference_steps: int = 100,
        guidance_scale: float = 3.5,
        negative_prompt: str | None = None,
    ) -> Tuple[np.ndarray, int]:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        final_negative_prompt = "Low quality." if negative_prompt is None else negative_prompt
        kwargs: dict[str, object] = {
            "prompt": prompt,
            "audio_length_in_s": float(seconds),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "num_waveforms_per_prompt": 1,
            "generator": generator,
            "negative_prompt": final_negative_prompt,
        }
        try:
            output = self.pipe(**kwargs)
        except TypeError:
            kwargs.pop("negative_prompt", None)
            output = self.pipe(**kwargs)
        audio_attr = getattr(output, "audio", None) or getattr(output, "audios", None)
        if audio_attr is None:
            raise RuntimeError("AudioLDM2Pipeline returned no audio output")
        audio_array = self._prepare_audio_array(audio_attr[0])
        return audio_array, TARGET_SAMPLE_RATE

    def generate_many(
        self,
        prompts: list[tuple[str, int, int, str | None]],
        num_inference_steps: int = 100,
        guidance_scale: float = 3.5,
    ) -> list[Tuple[np.ndarray, int]]:
        results: list[Tuple[np.ndarray, int]] = []
        for prompt, seconds, seed, negative_prompt in prompts:
            waveform, sr = self.generate(
                prompt,
                seconds,
                seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
            )
            results.append((waveform, sr))
        return results
