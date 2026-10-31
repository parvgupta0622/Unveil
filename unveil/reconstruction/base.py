"""Backend-agnostic image-to-3D interface.

Every backend returns meshes in the shared *view frame*: +x right, +y up, +z towards the camera that
"took" the input image. Verification, depth control and export only ever work in this frame, so a new
backend only has to implement this class (and register itself in `reconstruction/__init__.py`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import trimesh
from PIL import Image


class ReconstructionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraSpec:
    """Perspective camera on the +z axis of the view frame, looking at the origin."""

    distance: float
    fov_deg: float


@dataclass
class ReconstructionResult:
    mesh: trimesh.Trimesh  # view frame, with per-vertex colours when available
    handle: Any = None  # opaque backend state for later colour queries (texture baking)
    meta: Dict[str, Any] = field(default_factory=dict)


class Reconstructor(ABC):
    name: str = "base"
    camera: CameraSpec = CameraSpec(distance=2.0, fov_deg=40.0)

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def to(self, device: str) -> None: ...

    @abstractmethod
    def prepare_input(self, rgba: Image.Image, foreground_ratio: float) -> Image.Image:
        """Frame an RGBA cut-out the way this backend expects (padding, background colour, size)."""

    @abstractmethod
    def reconstruct(self, image: Image.Image, mc_resolution: int, mc_threshold: float) -> ReconstructionResult: ...

    @property
    def supports_color_query(self) -> bool:
        return False

    def query_color(self, handle: Any, points: np.ndarray) -> np.ndarray:
        """RGB in [0, 1] for view-frame points (N, 3). Used by the texture baker."""
        raise NotImplementedError(f"{self.name} cannot query surface colour")
