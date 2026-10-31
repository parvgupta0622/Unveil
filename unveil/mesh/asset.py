"""MeshAsset: a plain-array mesh container that survives disk round-trips and depth re-exports."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from PIL import Image


@dataclass
class MeshAsset:
    vertices: np.ndarray  # (N, 3) float, view frame: +x right, +y up, +z towards the input camera
    faces: np.ndarray  # (F, 3) int
    vertex_colors: Optional[np.ndarray] = None  # (N, 4) uint8
    uv: Optional[np.ndarray] = None  # (N, 2) float, OBJ convention (origin bottom-left)
    texture: Optional[np.ndarray] = None  # (H, W, 3) uint8

    @property
    def textured(self) -> bool:
        return self.uv is not None and self.texture is not None

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh) -> "MeshAsset":
        colors = None
        if mesh.visual.kind == "vertex":
            colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
        return cls(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int64), colors)

    def with_depth_scale(self, scale: float) -> "MeshAsset":
        """Scale geometry along the view (z) axis about its own centre. UVs/texture are unaffected."""
        v = self.vertices.astype(np.float64).copy()
        cz = 0.5 * (v[:, 2].max() + v[:, 2].min())
        v[:, 2] = cz + (v[:, 2] - cz) * float(scale)
        return replace(self, vertices=v.astype(np.float32))

    def centered(self) -> "MeshAsset":
        v = self.vertices.astype(np.float64)
        center = 0.5 * (v.max(axis=0) + v.min(axis=0))
        return replace(self, vertices=(v - center).astype(np.float32))

    def extents(self) -> np.ndarray:
        return self.vertices.max(axis=0) - self.vertices.min(axis=0)

    def to_trimesh(self, material: str = "pbr") -> trimesh.Trimesh:
        """material: 'pbr' for glTF (PBR base colour), 'simple' for OBJ/MTL."""
        if self.textured:
            image = Image.fromarray(self.texture)
            if material == "pbr":
                mat = trimesh.visual.material.PBRMaterial(
                    name="unveil", baseColorTexture=image, metallicFactor=0.0, roughnessFactor=0.85, doubleSided=True
                )
            else:
                mat = trimesh.visual.material.SimpleMaterial(name="unveil", image=image, diffuse=(255, 255, 255, 255))
            visual = trimesh.visual.TextureVisuals(uv=self.uv, material=mat)
            return trimesh.Trimesh(self.vertices, self.faces, visual=visual, process=False)
        return trimesh.Trimesh(self.vertices, self.faces, vertex_colors=self.vertex_colors, process=False)

    def save(self, path: Path) -> None:
        arrays = {"vertices": self.vertices, "faces": self.faces}
        for name in ("vertex_colors", "uv", "texture"):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "MeshAsset":
        with np.load(path) as data:
            return cls(**{k: data[k] for k in data.files})
