"""Reconstruction backend registry. Add new image-to-3D models here; TripoSR is the only one in this version."""
from __future__ import annotations

import importlib

from .base import CameraSpec, ReconstructionError, ReconstructionResult, Reconstructor

BACKENDS = {
    "triposr": "unveil.reconstruction.triposr:TripoSRReconstructor",
}


def create_reconstructor(name: str, **kwargs) -> Reconstructor:
    try:
        target = BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown reconstruction backend {name!r}; available: {sorted(BACKENDS)}") from None
    module_name, cls_name = target.split(":")
    return getattr(importlib.import_module(module_name), cls_name)(**kwargs)


__all__ = ["BACKENDS", "CameraSpec", "ReconstructionError", "ReconstructionResult", "Reconstructor", "create_reconstructor"]
