"""UV unwrapping (xatlas) and texture baking from a colour field, without OpenGL.

TripoSR's own baker needs a moderngl context, which is fragile on headless Colab. Here the UV atlas is
rasterised on the CPU with OpenCV, each texel is lifted back to 3D with barycentric coordinates, and the
reconstruction backend's colour field is queried at those points.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from scipy import ndimage as ndi

from .asset import MeshAsset

ColorFn = Callable[[np.ndarray], np.ndarray]  # (N, 3) view-frame points -> (N, 3) RGB in [0, 1]


def rasterize_face_ids(uvs: np.ndarray, faces: np.ndarray, resolution: int) -> np.ndarray:
    """Face index per texel (-1 where no face). Row 0 is the top of the image (v = 1)."""
    px = np.stack([uvs[:, 0] * resolution - 0.5, (1.0 - uvs[:, 1]) * resolution - 0.5], axis=-1)
    tris = np.round(px[faces] * 16).astype(np.int32)
    ids = np.arange(1, len(faces) + 1, dtype=np.int64)  # encode id+1 in 24 bits across 3 channels
    colors = np.stack([ids & 255, (ids >> 8) & 255, (ids >> 16) & 255], axis=-1).tolist()
    canvas = np.zeros((resolution, resolution, 3), np.uint8)
    for tri, color in zip(tris, colors):
        cv2.fillConvexPoly(canvas, tri, color, cv2.LINE_8, 4)
    decoded = canvas[..., 0].astype(np.int64) | (canvas[..., 1].astype(np.int64) << 8) | (canvas[..., 2].astype(np.int64) << 16)
    return decoded - 1


def barycentric(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-14, 1e-14, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bary = np.clip(np.stack([1.0 - v - w, v, w], axis=-1), 0.0, 1.0)  # texels on edges may fall just outside
    return bary / np.maximum(bary.sum(-1, keepdims=True), 1e-9)


def bake_texture(asset: MeshAsset, color_fn: ColorFn, resolution: int = 1024) -> MeshAsset:
    import xatlas

    vmapping, indices, uvs = xatlas.parametrize(asset.vertices.astype(np.float32), asset.faces.astype(np.uint32))
    vertices = asset.vertices[vmapping]
    faces = indices.astype(np.int64)
    uvs = uvs.astype(np.float64)

    face_id = rasterize_face_ids(uvs, faces, resolution)
    rows, cols = np.nonzero(face_id >= 0)
    fids = face_id[rows, cols]
    texel_uv = np.stack([(cols + 0.5) / resolution, 1.0 - (rows + 0.5) / resolution], axis=-1)
    bary = barycentric(texel_uv, uvs[faces[fids]])
    points = (bary[..., None] * vertices[faces[fids]].astype(np.float64)).sum(axis=1)

    texture = np.zeros((resolution, resolution, 3), np.uint8)
    texture[rows, cols] = np.clip(np.asarray(color_fn(points)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    # Fill empty texels with the nearest baked colour so bilinear filtering/mipmaps never sample black seams.
    empty = face_id < 0
    if empty.any() and (~empty).any():
        _, (ri, ci) = ndi.distance_transform_edt(empty, return_indices=True)
        texture = texture[ri, ci]

    colors = asset.vertex_colors[vmapping] if asset.vertex_colors is not None else None
    return MeshAsset(vertices.astype(np.float32), faces, colors, uvs.astype(np.float32), texture)
