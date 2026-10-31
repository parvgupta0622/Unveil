"""Small shared helpers: logging, timing, devices, image conversion, JSON."""
from __future__ import annotations

import contextlib
import json
import logging
import secrets
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

LOG = logging.getLogger("unveil")
if not LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[unveil] %(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    LOG.addHandler(_h)
    LOG.setLevel(logging.INFO)


class StageTimer:
    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round(self.timings.get(name, 0.0) + time.perf_counter() - start, 2)

    @property
    def total(self) -> float:
        return round(sum(self.timings.values()), 2)


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def resolve_device(preference: str = "auto") -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gpu_total_gb(device: str) -> float:
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def free_gpu_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def flatten_alpha(image: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """Composite RGBA/LA/P images onto a solid background and return RGB."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, background + (255,))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return image.convert("RGB")


def image_from_any(value: Any) -> Image.Image | None:
    """Accept PIL / numpy / path / Gradio ImageEditor dict and return a PIL image (or None)."""
    if value is None:
        return None
    if isinstance(value, dict):  # gr.Sketchpad / gr.ImageEditor value
        composite = value.get("composite")
        if composite is not None:
            return image_from_any(composite)
        layers = [image_from_any(layer) for layer in value.get("layers") or []]
        layers = [layer for layer in layers if layer is not None]
        background = image_from_any(value.get("background"))
        if background is None and not layers:
            return None
        base = (background or Image.new("RGBA", layers[0].size, (255, 255, 255, 0))).convert("RGBA")
        for layer in layers:
            base.alpha_composite(layer.convert("RGBA").resize(base.size))
        return base
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, np.ndarray):
        arr = value
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if isinstance(value, (str, Path)):
        return Image.open(value)
    raise TypeError(f"Unsupported image input type: {type(value)!r}")


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).convert("RGB")


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def to_jsonable(obj: Any) -> Any:
    return _jsonable(obj)


def write_json(obj: Any, path: Path) -> None:
    Path(path).write_text(json.dumps(_jsonable(obj), indent=2))
