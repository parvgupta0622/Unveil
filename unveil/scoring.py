"""Scoring of ControlNet variations: sketch-silhouette IoU + CLIP similarity + clean-background check."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from .config import ScoringConfig
from .utils import LOG

# Typical cosine ranges for CLIP ViT-B/32, used to map raw similarities onto [0, 1].
TEXT_COS_RANGE = (0.15, 0.35)
IMAGE_COS_RANGE = (0.40, 0.85)


@dataclass
class VariationScore:
    index: int
    iou: float
    clip: float
    clip_raw: float
    clean: float
    total: float
    passes_floor: bool
    clean_details: Dict[str, float] = field(default_factory=dict)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def clean_background_score(rgb: np.ndarray, alpha: np.ndarray) -> tuple[float, Dict[str, float]]:
    """How much the image looks like one clean object on a plain background (0..1)."""
    fg = alpha > 127
    frac = float(fg.mean())
    if frac < 0.08:
        area = frac / 0.08
    elif frac > 0.70:
        area = max(0.0, (0.95 - frac) / 0.25)
    else:
        area = 1.0

    n, _, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.zeros(0)
    single = float(areas.max() / areas.sum()) if areas.size else 0.0

    edge = np.concatenate([fg[:2].ravel(), fg[-2:].ravel(), fg[:, :2].ravel(), fg[:, -2:].ravel()])
    border = float(max(0.0, 1.0 - 5.0 * edge.mean()))

    bg_mask = ~(cv2.dilate(fg.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    if bg_mask.sum() < 100:
        uniform = 0.0
    else:
        std = float(rgb[bg_mask].astype(np.float32).std(axis=0).mean())
        uniform = float(np.exp(-std / 20.0))

    details = {"area": area, "single_object": single, "off_border": border, "plain_background": uniform}
    return float(np.mean(list(details.values()))), details


def _normalize(value: float, lo_hi: tuple[float, float]) -> float:
    lo, hi = lo_hi
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


class ClipScorer:
    def __init__(self, model_id: str, device: str, dtype) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._model = None
        self._processor = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        LOG.info("Loading CLIP %s", self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id, torch_dtype=self.dtype).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_id)

    def to(self, device: str) -> None:
        if self._model is not None:
            self._model.to(device)
        self.device = device

    @staticmethod
    def _features(out, *names):
        import torch

        if isinstance(out, torch.Tensor):
            return out
        for name in names:  # newer transformers may return a ModelOutput instead of a tensor
            value = getattr(out, name, None)
            if value is not None:
                return value
        raise TypeError(f"Unexpected CLIP output type {type(out)!r}")

    def image_embeddings(self, images: Sequence[Image.Image]):
        import torch

        self.load()
        inputs = self._processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        with torch.inference_mode():
            out = self._model.get_image_features(pixel_values=inputs["pixel_values"].to(self.device, self.dtype))
        feats = self._features(out, "image_embeds", "pooler_output").float()
        return torch.nn.functional.normalize(feats, dim=-1)

    def text_embeddings(self, texts: Sequence[str]):
        import torch

        self.load()
        inputs = self._processor(text=list(texts), padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            out = self._model.get_text_features(
                input_ids=inputs["input_ids"].to(self.device), attention_mask=inputs["attention_mask"].to(self.device)
            )
        feats = self._features(out, "text_embeds", "pooler_output").float()
        return torch.nn.functional.normalize(feats, dim=-1)

    def similarity(self, images: Sequence[Image.Image], prompt: str, reference: Optional[Image.Image]) -> List[tuple[float, float]]:
        """(normalized, raw) cosine per image: against the prompt, or the sketch itself when no prompt."""
        img = self.image_embeddings(images)
        if prompt.strip():
            ref = self.text_embeddings([f"a photo of {prompt.strip()}"])
            rng = TEXT_COS_RANGE
        else:
            ref = self.image_embeddings([reference])
            rng = IMAGE_COS_RANGE
        raw = (img @ ref.T).squeeze(-1).cpu().numpy()
        return [(_normalize(float(r), rng), float(r)) for r in raw]


def score_variations(
    images: Sequence[Image.Image],
    alphas: Sequence[np.ndarray],
    sketch_silhouette: np.ndarray,
    clip_scores: Sequence[tuple[float, float]],
    cfg: ScoringConfig,
) -> List[VariationScore]:
    results = []
    for i, (image, alpha, (clip_n, clip_raw)) in enumerate(zip(images, alphas, clip_scores)):
        mask = cv2.resize((alpha > 127).astype(np.uint8), sketch_silhouette.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        iou = mask_iou(mask, sketch_silhouette)
        clean, details = clean_background_score(np.asarray(image.convert("RGB")), alpha)
        total = cfg.w_iou * iou + cfg.w_clip * clip_n + cfg.w_clean * clean
        results.append(
            VariationScore(i, round(iou, 4), round(clip_n, 4), round(clip_raw, 4), round(clean, 4), round(total, 4), iou >= cfg.iou_floor, {k: round(v, 3) for k, v in details.items()})
        )
    return results


def rank_variations(scores: Sequence[VariationScore]) -> List[int]:
    """Indices best-first. Variations that drift from the sketch structure (IoU below floor) rank last."""
    return [s.index for s in sorted(scores, key=lambda s: (s.passes_floor, s.total), reverse=True)]
