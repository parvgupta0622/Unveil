"""Hollow vs solid interiors.

TripoSR (like most single-image reconstructors) always returns a closed solid: a cup comes back as a filled
cylinder. For vessels (cups, bowls, bottles, vases, ...) this module carves a real cavity:

1. Decide hollow vs solid from the object's name (prompt keywords) or, without one, a CLIP zero-shot guess.
2. Voxelise the mesh as a solid (ray parity along all three axes, majority vote: robust to small holes).
3. Fill each horizontal slice so a partial dip TripoSR may have made becomes solid again.
4. Erode by the wall thickness (Euclidean distance transform) -> cavity with equal walls and floor.
5. Drill the cavity open through the top (+y), following the rim/neck slice by slice.
6. Marching cubes on the remaining shell; colours come from the nearest point of the original surface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .processing import signed_volume, smooth_mesh

# Vessels: objects whose interior is normally empty and open at the top.
HOLLOW_OBJECTS = (
    "cup", "mug", "teacup", "coffee cup", "bowl", "vase", "bottle", "glass", "wine glass", "tumbler", "goblet",
    "chalice", "jar", "pot", "flower pot", "flowerpot", "planter", "bucket", "pail", "pitcher", "jug", "can",
    "beaker", "flask", "urn", "basket", "bin", "trash can", "pen holder", "tankard", "carafe", "decanter",
)
SOLID_OBJECTS = (
    "chair", "table", "stool", "bench", "sofa", "couch", "desk", "bed", "lamp", "ball", "sphere", "cube",
    "block", "pyramid", "cone", "cylinder", "car", "truck", "airplane", "plane", "rocket", "robot", "toy",
    "figure", "figurine", "statue", "sculpture", "character", "person", "animal", "cat", "dog", "bird",
    "fish", "horse", "duck", "bear", "teddy bear", "house", "building", "tower", "tree", "mushroom", "cactus",
    "apple", "fruit", "banana", "rock", "stone", "shoe", "phone", "book", "sword", "key", "guitar",
    "candle", "light bulb", "bulb", "crown", "gem", "crystal", "heart", "star", "donut", "cake", "burger",
    "chess piece", "pawn", "trophy", "helmet", "hammer", "spoon", "fork", "knife", "skull",
)
# Words that end the noun phrase naming the object ("a lamp with a glass shade" -> "a lamp").
_PHRASE_BREAK = re.compile(r"\b(?:with|of|on|in|and|for|from|containing|holding|that|which|who|made)\b|[,;.]")


@dataclass
class InteriorDecision:
    hollow: bool
    label: str  # object category that decided it ("" if unknown)
    source: str  # "user" | "prompt" | "clip" | "default"
    confidence: float = 1.0
    top_labels: List[Tuple[str, float]] = field(default_factory=list)

    def describe(self) -> str:
        kind = "Hollow" if self.hollow else "Solid"
        if self.source == "user":
            return f"{kind} (your choice)"
        if self.source == "prompt":
            return f"{kind} — '{self.label}' found in your description"
        if self.source == "clip":
            return f"{kind} — CLIP recognises a '{self.label}' (p(hollow) = {self.confidence:.2f})"
        return f"{kind} (no object category recognised)"


def _find_category(text: str) -> Optional[str]:
    """Head noun of the first noun phrase that names a known category (longest match wins at a position)."""
    vocab = sorted(set(HOLLOW_OBJECTS + SOLID_OBJECTS), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in vocab) + r")(?:e?s)?\b")
    text = text.lower()
    head = _PHRASE_BREAK.split(text, maxsplit=1)[0]
    for chunk in (head, text):
        found = [m.group(1) for m in pattern.finditer(chunk)]
        if found:
            return found[-1] if chunk is head else found[0]  # English head noun = last noun of the phrase
    return None


def decide_interior(
    prompt: str,
    override: str = "auto",
    zero_shot: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
) -> InteriorDecision:
    """override: 'auto' | 'hollow' | 'solid'. zero_shot(labels) -> probabilities for the generated image."""
    if override in ("hollow", "solid"):
        return InteriorDecision(override == "hollow", "", "user")
    label = _find_category(prompt or "")
    if label is not None:
        return InteriorDecision(label in HOLLOW_OBJECTS, label, "prompt")
    if zero_shot is not None:
        labels = list(HOLLOW_OBJECTS) + list(SOLID_OBJECTS)
        probs = np.asarray(zero_shot(labels), dtype=np.float64)
        p_hollow = float(probs[: len(HOLLOW_OBJECTS)].sum())
        order = np.argsort(-probs)[:3]
        top = [(labels[i], round(float(probs[i]), 3)) for i in order]
        return InteriorDecision(p_hollow >= 0.5, top[0][0], "clip", round(p_hollow, 3), top)
    return InteriorDecision(False, "", "default")


# ---------------------------------------------------------------------------------------------- voxels
def voxelize_solid(vertices: np.ndarray, faces: np.ndarray, origin: np.ndarray, pitch: float, shape: Tuple[int, int, int]) -> np.ndarray:
    """Inside/outside per voxel centre. Rays along each axis count surface crossings (even-odd rule); a voxel is
    inside when most axes with a consistent (even) crossing count agree, so a few bad rays cannot leak."""
    tri = (np.asarray(vertices, np.float64)[faces] - origin) / pitch - 0.5  # voxel-centre index space
    shape = tuple(int(s) for s in shape)
    votes = np.zeros(shape, np.int8)
    valid = np.zeros(shape, np.int8)
    for w in range(3):
        u, v = [a for a in range(3) if a != w]
        nu, nv, nw = shape[u], shape[v], shape[w]
        pu, pv, pw = tri[..., u] + 1.37e-4, tri[..., v] + 2.71e-4, tri[..., w]  # jitter rays off shared edges
        i0 = np.clip(np.ceil(pu.min(1)), 0, nu).astype(np.int64)
        i1 = np.clip(np.floor(pu.max(1)), -1, nu - 1).astype(np.int64)
        j0 = np.clip(np.ceil(pv.min(1)), 0, nv).astype(np.int64)
        j1 = np.clip(np.floor(pv.max(1)), -1, nv - 1).astype(np.int64)
        cu, cv = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
        n = cu * cv
        if n.sum() == 0:
            continue
        t = np.repeat(np.arange(len(tri)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        ri, rj = i0[t] + k % cu[t], j0[t] + k // cu[t]
        # 2D barycentric test of the ray (ri, rj) against the projected triangle
        au, av, bu, bv, cu_, cv_ = pu[t, 0], pv[t, 0], pu[t, 1], pv[t, 1], pu[t, 2], pv[t, 2]
        den = (bv - cv_) * (au - cu_) + (cu_ - bu) * (av - cv_)
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        l0 = ((bv - cv_) * (ri - cu_) + (cu_ - bu) * (rj - cv_)) / den
        l1 = ((cv_ - av) * (ri - cu_) + (au - cu_) * (rj - cv_)) / den
        hit = ok & (l0 >= 0) & (l1 >= 0) & (l0 + l1 <= 1)
        t, ri, rj, l0, l1 = t[hit], ri[hit], rj[hit], l0[hit], l1[hit]
        depth = l0 * pw[t, 0] + l1 * pw[t, 1] + (1 - l0 - l1) * pw[t, 2]

        col = ri * nv + rj
        order = np.lexsort((depth, col))
        col, depth = col[order], depth[order]
        counts = np.bincount(col, minlength=nu * nv)
        rank = np.arange(len(col)) - np.repeat(np.cumsum(counts) - counts, counts)[: len(col)]
        even = counts % 2 == 0
        enter = (rank % 2 == 0) & even[col]
        k0 = np.clip(np.ceil(depth[enter]), 0, nw).astype(np.int64)
        k1 = np.clip(np.floor(depth[np.flatnonzero(enter) + 1]), -1, nw - 1).astype(np.int64) + 1
        diff = np.zeros((nu * nv, nw + 1), np.int32)
        np.add.at(diff, (col[enter], k0), 1)
        np.add.at(diff, (col[enter], np.maximum(k1, k0)), -1)
        inside = (np.cumsum(diff, axis=1)[:, :nw] > 0).reshape(nu, nv, nw)
        axes = [0, 0, 0]
        axes[u], axes[v], axes[w] = 0, 1, 2
        votes += np.transpose(inside, axes).astype(np.int8)
        valid += np.transpose(np.broadcast_to(even.reshape(nu, nv, 1), (nu, nv, nw)), axes).astype(np.int8)
    return (votes * 2 > valid) & (valid > 0)


def _largest(mask: np.ndarray) -> np.ndarray:
    labels, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


@dataclass
class HollowStats:
    wall_thickness: float  # model units
    wall_fraction: float  # of the object's largest dimension
    voxel_pitch: float
    opened: bool  # cavity reaches the outside through the top
    cavity_fraction: float  # cavity volume / filled volume
    faces: int

    def describe(self) -> str:
        top = "open at the top" if self.opened else "closed (sealed cavity)"
        return f"cavity {self.cavity_fraction:.0%} of the volume, walls ≈ {self.wall_fraction:.1%} of the size, {top}"


@dataclass
class HollowResult:
    mesh: trimesh.Trimesh
    stats: HollowStats
    remap: Callable[[np.ndarray], np.ndarray]  # texture-bake points -> where to sample the colour field


def hollow_out(mesh: trimesh.Trimesh, wall_fraction: float = 0.05, resolution: int = 160, smooth_iterations: int = 3) -> Tuple[Optional[HollowResult], str]:
    """Carve an open-top cavity with walls `wall_fraction` x the object's size. Returns (result, reason)."""
    vertices = np.asarray(mesh.vertices, np.float64)
    lo, hi = vertices.min(0), vertices.max(0)
    pitch = float((hi - lo).max()) / resolution
    pad = 3
    origin = lo - pad * pitch
    shape = tuple(np.ceil((hi - lo) / pitch).astype(int) + 2 * pad + 1)
    solid = voxelize_solid(vertices, np.asarray(mesh.faces), origin, pitch, shape)
    solid = _largest(ndi.binary_closing(solid, iterations=1) | solid)
    if solid.sum() < 1000:
        return None, "the mesh could not be voxelised as a closed solid"

    # Fill each horizontal (x-z) slice: partial dips become solid, the cavity is then cut with even walls.
    filled = np.stack([ndi.binary_fill_holes(solid[:, j, :]) for j in range(shape[1])], axis=1)
    t_vox = max(1.5, wall_fraction * resolution)
    cavity = ndi.distance_transform_edt(filled) > t_vox
    # Opening drops thin channels of cavity, e.g. inside a mug handle or along a narrow neck (the neck is
    # drilled separately below), so only the vessel body is hollowed.
    r = max(2, int(round(0.5 * t_vox)))
    zz, yy, xx = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    cavity = _largest(ndi.binary_opening(cavity, structure=(xx**2 + yy**2 + zz**2) <= r * r))
    if cavity.sum() < 0.02 * filled.sum():
        return None, "the object is too thin for walls of this thickness"

    # Drill through the top: follow the cavity's top footprint upward slice by slice (handles necks and rims).
    mouth = np.zeros_like(filled)
    top = int(np.nonzero(cavity.any(axis=(0, 2)))[0].max())
    footprint = cavity[:, top, :]
    opened = False
    for j in range(top + 1, shape[1]):
        grown = cv2.dilate(footprint.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        layer = filled[:, j, :]
        if not (layer & grown).any():
            opened = True
            break
        d2 = ndi.distance_transform_edt(layer)
        footprint = (d2 > min(t_vox, 0.5 * d2[grown].max())) & grown
        if not footprint.any():
            break
        mouth[:, j, :] = footprint
    shell = _largest(filled & ~cavity & ~mouth)

    field = ndi.gaussian_filter(shell.astype(np.float32), 0.8)
    from skimage.measure import marching_cubes

    verts, faces, _, _ = marching_cubes(field, 0.5)
    verts = verts * pitch + origin
    if signed_volume(verts, faces) < 0:
        faces = faces[:, ::-1]

    tree = cKDTree(vertices)
    colors = None
    if mesh.visual.kind == "vertex":
        _, idx = tree.query(verts)
        colors = np.asarray(mesh.visual.vertex_colors)[idx]
    out = smooth_mesh(trimesh.Trimesh(verts, np.ascontiguousarray(faces), vertex_colors=colors, process=False), smooth_iterations)

    def remap(points: np.ndarray) -> np.ndarray:
        # Outer surface: sample the colour field where the point is. Inner walls: borrow the colour of the
        # closest point of the original surface instead of the field's (undefined) interior.
        dist, idx = tree.query(points)
        return np.where((dist < 1.5 * pitch)[:, None], points, vertices[idx])

    stats = HollowStats(
        wall_thickness=round(t_vox * pitch, 5),
        wall_fraction=round(t_vox / resolution, 4),
        voxel_pitch=round(pitch, 5),
        opened=opened,
        cavity_fraction=round(float((cavity | mouth).sum() / filled.sum()), 4),
        faces=int(len(out.faces)),
    )
    return HollowResult(out, stats, remap), ""


def section_image(vertices: np.ndarray, faces: np.ndarray, size: int = 256, color: Tuple[int, int, int] = (20, 150, 140)) -> Image.Image:
    """Front-to-back cut through the middle of the model (x-y plane at the centre depth): shows walls/cavity."""
    v = np.asarray(vertices, np.float64)
    lo, hi = v.min(0), v.max(0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1])) * 1.1
    pitch = span / size
    center = 0.5 * (lo + hi)
    origin = np.array([center[0] - span / 2, center[1] - span / 2, center[2] - pitch / 2])
    inside = voxelize_solid(v, np.asarray(faces), origin, pitch, (size, size, 1))[:, :, 0]
    img = np.full((size, size, 3), 248, np.uint8)
    mask = np.flipud(inside.T)  # rows = -y (image up = +y), cols = x
    img[mask] = color
    edge = mask & ~ndi.binary_erosion(mask)
    img[edge] = (np.array(color) * 0.55).astype(np.uint8)
    return Image.fromarray(img)
