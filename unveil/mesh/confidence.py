"""How sure is the model? A per-point confidence map of the final surface, shown as a heatmap.

Two signals, both measured rather than predicted:

* Agreement. Every run makes several TripoSR attempts (two framings per chosen variation). They are aligned
  on the front view (width/height and the front surface) and, for each point of the chosen mesh, we measure
  how far the other attempts' surfaces are. Where they all land in the same place the shape is well
  determined; where they spread out, TripoSR was guessing.
* Visibility. Points the input camera could see were constrained by the image; hidden ones were imagined.

A point whose position a guide view's outline fixes (see guide.py) counts as mostly known.
Score 0 = sure (green) ... 1 = a guess (red).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import trimesh
from scipy import sparse
from scipy.spatial import cKDTree

from ..reconstruction.base import CameraSpec
from .asset import MeshAsset
from .verify import project

SURE, GUESS = 1 / 3, 2 / 3  # thresholds of the three colour bands
_COLOURS = np.array([(35, 170, 90), (245, 200, 50), (220, 60, 50)], np.float64)  # green, yellow, red


@dataclass
class ConfidenceStats:
    attempts: int  # other TripoSR attempts compared against
    seen: float  # share of the surface the input camera saw
    sure: float  # share rated sure (green)
    guess: float  # share rated a guess (red)
    guided: float  # share fixed by a guide view's outline

    def describe(self) -> str:
        basis = f"{self.attempts} other attempt{'s' if self.attempts != 1 else ''} compared" if self.attempts else "visibility only (one attempt)"
        out = f"{self.sure:.0%} sure · {self.guess:.0%} guessed · {self.seen:.0%} seen in the image ({basis}"
        return out + (f", {self.guided:.0%} fixed by your guide views)" if self.guided else ")")


@dataclass
class ConfidenceField:
    points: np.ndarray  # (N, 3) points on the scored surface
    values: np.ndarray  # (N,) 0 = sure ... 1 = guess
    stats: ConfidenceStats

    def at(self, points: np.ndarray) -> np.ndarray:
        _, idx = cKDTree(self.points).query(np.asarray(points, np.float64))
        return self.values[idx]


def _align(vertices: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match another attempt to the reference by its front-view box (x/y size and centre) and front surface."""
    lo, hi = vertices.min(0), vertices.max(0)
    rlo, rhi = reference.min(0), reference.max(0)
    scale = float(np.mean((rhi[:2] - rlo[:2]) / np.maximum(hi[:2] - lo[:2], 1e-9)))
    out = (vertices - np.r_[0.5 * (lo[:2] + hi[:2]), np.percentile(vertices[:, 2], 99.5)]) * scale
    return out + np.r_[0.5 * (rlo[:2] + rhi[:2]), np.percentile(reference[:, 2], 99.5)]


def visible_points(mesh: trimesh.Trimesh, camera: CameraSpec, size: int = 128, tolerance: float = 0.03) -> np.ndarray:
    """Vertices the input camera sees: facing it, and within `tolerance` x object size of the nearest surface
    along their pixel (so parts hidden behind other parts do not count)."""
    v = np.asarray(mesh.vertices, np.float64)
    to_cam = np.array([0.0, 0.0, camera.distance]) - v
    facing = np.einsum("ij,ij->i", np.asarray(mesh.vertex_normals), to_cam) > 0.05 * np.linalg.norm(to_cam, axis=1)
    depth = camera.distance - v[:, 2]
    px = np.clip(np.floor(project(v, camera, size) + 0.5).astype(np.int64), 0, size - 1)
    flat = px[:, 1] * size + px[:, 0]
    zbuf = np.full(size * size, np.inf)
    np.minimum.at(zbuf, flat, depth)
    extent = float((v.max(0) - v.min(0)).max())
    return facing & (depth <= zbuf[flat] + tolerance * extent)


def _smooth(values: np.ndarray, faces: np.ndarray, iterations: int) -> np.ndarray:
    n = len(values)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adj = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.float64)
    deg = np.asarray(adj.sum(axis=1)).ravel() + 1.0
    for _ in range(iterations):
        values = (adj @ values + values) / deg
    return values


def surface_confidence(
    shape: trimesh.Trimesh,
    reference: trimesh.Trimesh,
    attempts: Sequence[trimesh.Trimesh],
    camera: CameraSpec,
    pinned: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    spread: float = 0.05,
    smooth_iterations: int = 4,
) -> ConfidenceField:
    """Confidence of every vertex of `shape` (the chosen mesh after guide trimming), with `reference` the
    TripoSR mesh it came from and `attempts` the other TripoSR meshes of the run. `spread` is the disagreement
    (x object size) that counts as a full guess."""
    points = np.asarray(shape.vertices, np.float64)
    ref = np.asarray(reference.vertices, np.float64)
    extent = float((ref.max(0) - ref.min(0)).max())
    hidden = ~visible_points(shape, camera)
    if attempts:
        dist = np.zeros(len(points))
        for other in attempts:
            aligned = _align(np.asarray(other.vertices, np.float64), ref)
            dist += cKDTree(aligned).query(points)[0]
        disagreement = np.clip(dist / len(attempts) / (spread * extent), 0.0, 1.0)
        values = 0.65 * disagreement + 0.35 * hidden
    else:
        values = 0.5 * hidden.astype(np.float64)
    fixed = pinned(points) if pinned is not None else np.zeros(len(points), bool)
    values = np.where(fixed, 0.4 * values, values)
    values = np.clip(_smooth(values, np.asarray(shape.faces), smooth_iterations), 0.0, 1.0)
    stats = ConfidenceStats(
        attempts=len(attempts),
        seen=round(float(1.0 - hidden.mean()), 3),
        sure=round(float((values < SURE).mean()), 3),
        guess=round(float((values > GUESS).mean()), 3),
        guided=round(float(fixed.mean()), 3),
    )
    return ConfidenceField(points, values.astype(np.float32), stats)


def heat_colors(values: np.ndarray) -> np.ndarray:
    """0 -> green, 0.5 -> yellow, 1 -> red, as RGBA uint8."""
    t = np.clip(np.asarray(values, np.float64), 0.0, 1.0) * 2.0
    i = np.minimum(t.astype(int), 1)
    rgb = _COLOURS[i] + (_COLOURS[i + 1] - _COLOURS[i]) * (t - i)[:, None]
    return np.c_[np.round(rgb), np.full(len(rgb), 255)].astype(np.uint8)


def save_confidence(path: Path, points: np.ndarray, values: np.ndarray) -> None:
    np.savez_compressed(path, points=np.asarray(points, np.float32), values=np.asarray(values, np.float32))


def export_heatmap(asset: MeshAsset, confidence_path: Path, obj_path: Path, depth_scale: float = 1.0) -> Optional[Path]:
    """OBJ with the confidence as vertex colours, scaled and centred like the model's own export. Values are
    looked up by nearest point, so they survive a later texture bake (which re-indexes the vertices)."""
    confidence_path = Path(confidence_path)
    if not confidence_path.exists():
        return None
    with np.load(confidence_path) as data:
        _, idx = cKDTree(data["points"]).query(asset.vertices)
        values = data["values"][idx]
    shaped = MeshAsset(asset.vertices, asset.faces).with_depth_scale(depth_scale).centered()
    rgb = heat_colors(values)[:, :3] / 255.0
    lines = [f"v {x:.6f} {y:.6f} {z:.6f} {r:.3f} {g:.3f} {b:.3f}" for (x, y, z), (r, g, b) in zip(shaped.vertices, rgb)]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in shaped.faces]
    Path(obj_path).write_text("# Unveil confidence heatmap: green = sure, yellow = partly guessed, red = guessed\n" + "\n".join(lines) + "\n")
    return Path(obj_path)
