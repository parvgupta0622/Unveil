"""Sketch-guided image generation with Stable Diffusion 1.5 + ControlNet v1.1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from PIL import Image

from .config import CreativityLevel, GenerationConfig
from .utils import LOG, load_without_empty_weights


@dataclass
class Variation:
    index: int
    level: CreativityLevel
    seed: int
    image: Image.Image


def build_prompt(cfg: GenerationConfig, user_prompt: str) -> str:
    subject = (user_prompt or "").strip().rstrip(".,") or cfg.default_subject
    return cfg.prompt_template.format(subject=subject)


class SketchImageGenerator:
    """Owns one SD1.5 pipeline and swaps ControlNets in and out of it per control mode."""

    def __init__(self, cfg: GenerationConfig, device: str, dtype) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self._pipe = None
        self._mode: Optional[str] = None
        self._controlnets: Dict[str, object] = {}

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def _from_pretrained(self, cls, repo: str, **kwargs):
        return load_without_empty_weights(lambda **extra: self._load(cls, repo, **kwargs, **extra), repo)

    def _load(self, cls, repo: str, **kwargs):
        import torch

        if self.dtype == torch.float16:
            try:
                return cls.from_pretrained(repo, torch_dtype=self.dtype, variant="fp16", **kwargs)
            except (OSError, ValueError):
                pass  # repo has no fp16 variant files; load full weights and cast
        return cls.from_pretrained(repo, torch_dtype=self.dtype, **kwargs)

    def _controlnet(self, mode: str):
        from diffusers import ControlNetModel

        if mode not in self.cfg.controlnets:
            raise ValueError(f"Unknown control mode {mode!r}; expected one of {list(self.cfg.controlnets)}")
        if mode not in self._controlnets:
            LOG.info("Loading ControlNet %s", self.cfg.controlnets[mode])
            self._controlnets[mode] = self._from_pretrained(ControlNetModel, self.cfg.controlnets[mode]).to(self.device)
        return self._controlnets[mode]

    def load(self, mode: str = "precise"):
        from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler

        controlnet = self._controlnet(mode)
        if self._pipe is None:
            LOG.info("Loading base model %s", self.cfg.base_model)
            pipe = self._from_pretrained(
                StableDiffusionControlNetPipeline,
                self.cfg.base_model,
                controlnet=controlnet,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.set_progress_bar_config(disable=True)
            self._pipe = pipe.to(self.device)
        elif self._mode != mode:
            self._pipe.register_modules(controlnet=controlnet)
        self._mode = mode
        return self._pipe

    def to(self, device: str) -> None:
        if self._pipe is not None:
            self._pipe.to(device)
        for cn in self._controlnets.values():
            cn.to(device)
        self.device = device

    def generate(
        self,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        level: CreativityLevel,
        seed: int,
        mode: str,
        steps: int,
        guidance_scale: float,
        step_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Image.Image:
        import torch

        pipe = self.load(mode)
        kwargs = {}
        if step_callback is not None:

            def _on_step(_pipe, step, _t, cb_kwargs):
                step_callback(step + 1, steps)
                return cb_kwargs

            kwargs["callback_on_step_end"] = _on_step
        size = self.cfg.resolution
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image.resize((size, size)),
                height=size,
                width=size,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=level.conditioning_scale,
                control_guidance_end=level.guidance_end,
                # CPU generator => identical results for a given seed regardless of GPU type.
                generator=torch.Generator("cpu").manual_seed(int(seed)),
                **kwargs,
            )
        return result.images[0]
