"""Front-back symmetry: rebuild the hidden back of the model as a mirror image of the visible front.

A single image only shows the front; TripoSR has to guess the back. When the user says the object is
front-back symmetric (it looks the same from behind: a vase, a car or a mug seen from the side, ...), the
guess can be replaced by a reflection of what was actually seen:

1. Find the mirror plane z = z0 (perpendicular to the viewing direction, +z = towards the camera). For a
   symmetric solid every front-to-back column through it has its midpoint on that plane; z0 is the median
   midpoint of the columns along the object's outline, where the back TripoSR guessed matters least.
2. Clip the mesh exactly at z0 (triangles crossing the plane are split, so no detail is lost), keep the front
   half and add its reflection. Cut vertices lie on the plane and are shared by both halves: the result is
   closed wherever the input was.
3. Colours: the back half reuses the front's colours (vertex colours here, the colour field via `remap` for
   texture baking).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import trimesh
from scipy import ndimage as ndi

from .hollow import voxelize_solid
from .processing import signed_volume


@dataclass
class SymmetryStats:
    plane_z: float  # mirror plane depth in the view frame (+z = towards the camera)
    depth_before: float  # depth extent / max(width, height)
    depth_after: float
    faces: int

    def describe(self) -> str:
        return f"back rebuilt as a mirror of the front (depth {self.depth_before:.2f} → {self.depth_after:.2f} × width)"


@dataclass
class SymmetryResult:
    mesh: trimesh.Trimesh
    stats: SymmetryStats
    remap: Callable[[np.ndarray], np.ndarray]  # texture-bake points -> where to sample the colour field


def symmetry_plane(vertices: np.ndarray, faces: np.ndarray, resolution: int = 128) -> Optional[float]:
    """Depth of the front-back mirror plane: median midpoint of the solid's front-to-back columns at its outline."""
    lo, hi = vertices.min(0), vertices.max(0)
    pitch = float((hi - lo).max()) / resolution
    origin = lo - 2 * pitch
    shape = tuple(np.ceil((hi - lo) / pitch).astype(int) + 5)
    solid = voxelize_solid(vertices, faces, origin, pitch, shape)
    columns = solid.any(axis=2)
    if columns.sum() < 20:
        return None
    # Only the outline: there front and back meet on the plane, and the outline is what the input image
    # pins down; the inner columns' back halves are exactly the part TripoSR had to guess.
    band = columns & ~ndi.binary_erosion(columns, iterations=3)
    if band.sum() < 10:
        band = columns
    k = np.arange(shape[2])
    front = np.where(solid, k, -1).max(axis=2)[band]
    back = np.where(solid, k, shape[2]).min(axis=2)[band]
    return float(origin[2] + (np.median(0.5 * (front + back)) + 0.5) * pitch)


def _depth_ratio(vertices: np.ndarray) -> float:
    ext = vertices.max(0) - vertices.min(0)
    return float(ext[2] / max(ext[0], ext[1], 1e-9))


def clip_front(vertices: np.ndarray, faces: np.ndarray, z0: float, colors: Optional[np.ndarray] = None):
    """Part of the mesh with z >= z0, crossing triangles split at the plane. Returns (vertices, faces, colors,
    on_plane mask). Cut points on a shared edge are created once, so neighbouring triangles stay connected."""
    v = np.asarray(vertices, np.float64).copy()
    s = v[:, 2] - z0
    eps = 1e-7 * float(np.ptp(v, axis=0).max())
    s[np.abs(s) < eps] = 0.0
    v[s == 0.0, 2] = z0
    front = s >= 0
    fs = s[faces]
    keep = front[faces].all(axis=1)
    crossing = (fs > 0).any(axis=1) & (fs < 0).any(axis=1)

    new_v, new_c = [], []
    cut_index: Dict[Tuple[int, int], int] = {}
    col = None if colors is None else np.asarray(colors, np.float64)

    def cut(a: int, b: int) -> int:  # point on edge (a, b) where it crosses the plane
        if s[a] == 0.0:
            return a
        if s[b] == 0.0:
            return b
        key = (min(a, b), max(a, b))
        if key not in cut_index:
            t = s[a] / (s[a] - s[b])
            p = v[a] + t * (v[b] - v[a])
            p[2] = z0
            cut_index[key] = len(v) + len(new_v)
            new_v.append(p)
            if col is not None:
                new_c.append(col[a] + t * (col[b] - col[a]))
        return cut_index[key]

    out = [faces[keep]]
    extra = []
    for tri in faces[crossing]:
        inside = front[tri]
        r = int(np.argmax(inside & ~np.roll(inside, 1)))  # rotate so the front vertices come first (winding kept)
        a, b, c = np.roll(tri, -r)
        if inside[(r + 1) % 3]:  # two front vertices a, b; c behind
            bc, ca = cut(b, c), cut(c, a)
            extra += [(a, b, bc), (a, bc, ca)]
        else:  # one front vertex a; b, c behind
            extra.append((a, cut(a, b), cut(c, a)))
    if extra:
        out.append(np.asarray(extra, dtype=faces.dtype))
    f = np.concatenate(out)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    if new_v:
        v = np.vstack([v, np.asarray(new_v)])
        if col is not None:
            col = np.vstack([col, np.asarray(new_c)])
    on_plane = v[:, 2] == z0

    used, inverse = np.unique(f.ravel(), return_inverse=True)
    return v[used], inverse.reshape(-1, 3), (None if col is None else col[used]), on_plane[used]


def mirror_front_back(mesh: trimesh.Trimesh, resolution: int = 128) -> Tuple[Optional[SymmetryResult], str]:
    """Replace the back half by a reflection of the front half. Returns (result, reason)."""
    vertices, faces = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces)
    z0 = symmetry_plane(vertices, faces, resolution)
    if z0 is None:
        return None, "the mesh could not be voxelised as a closed solid"
    colors = np.asarray(mesh.visual.vertex_colors) if mesh.visual.kind == "vertex" else None
    fv, ff, fc, on_plane = clip_front(vertices, faces, z0, colors)
    if len(ff) < 0.2 * len(faces):
        return None, "too little of the model lies in front of the mirror plane"

    # Reflection z -> 2 z0 - z with reversed winding; vertices on the plane are shared, not duplicated.
    n = len(fv)
    back_ids = np.where(on_plane, np.arange(n), n + np.cumsum(~on_plane) - 1)
    mirrored = fv[~on_plane].copy()
    mirrored[:, 2] = 2 * z0 - mirrored[:, 2]
    out_v = np.vstack([fv, mirrored])
    out_f = np.vstack([ff, back_ids[ff][:, ::-1]])
    out_c = None if fc is None else np.vstack([fc, fc[~on_plane]]).round().astype(np.uint8)
    if signed_volume(out_v, out_f) < 0:
        out_f = out_f[:, ::-1]
    out = trimesh.Trimesh(out_v, out_f, vertex_colors=out_c, process=False)

    def remap(points: np.ndarray) -> np.ndarray:
        p = np.array(points, np.float64, copy=True)
        behind = p[:, 2] < z0
        p[behind, 2] = 2 * z0 - p[behind, 2]
        return p

    stats = SymmetryStats(round(z0, 5), round(_depth_ratio(vertices), 3), round(_depth_ratio(out_v), 3), int(len(out_f)))
    return SymmetryResult(out, stats, remap), ""
