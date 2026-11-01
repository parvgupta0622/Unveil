"""Guide views: use extra evidence as a scaffold around TripoSR's mesh instead of rebuilding it.

TripoSR only sees the front, so the depth and the back are guesses, and it often closes gaps it should have
left open (the opening of a mug handle). Two kinds of evidence correct that without throwing its detail away:

* Optional side / top sketches. Each one is filled into a silhouette and pushed through the mesh like a
  cookie cutter (orthographic, along the x or y axis): material outside it is trimmed, nothing is added.
  The sketch is registered to the mesh by the front view, which is the part TripoSR got right: its height
  (side) or width (top) sets the scale and the front edge sets the position.
* See-through holes. A region that is background in the generated image AND an enclosed loop in the user's
  sketch is a hole through the object (two sources must agree, so a bright highlight is never drilled).
  Voxels whose camera ray passes through it are removed.

Cutting runs on a voxel copy of the mesh (as hollowing does); colours come from the nearest original point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from ..reconstruction.base import CameraSpec
from .hollow import voxelize_solid
from .processing import signed_volume, smooth_mesh
from .verify import project, render_silhouette


@dataclass
class GuideStats:
    views: List[str]  # guide sketches used: "side", "top"
    trimmed_fraction: float  # share of TripoSR's volume outside the guide views (cut away)
    holes: int  # see-through holes opened
    hole_fraction: float  # share of the volume drilled away for them
    depth_before: float  # depth (z extent) / max(width, height)
    depth_after: float
    faces: int

    def describe(self) -> str:
        parts = []
        if self.views:
            views = " and ".join(f"{v} view" for v in self.views)
            if self.trimmed_fraction >= 0.005:
                parts.append(f"trimmed to your {views} ({self.trimmed_fraction:.0%} of the volume cut, depth {self.depth_before:.2f} → {self.depth_after:.2f} × size)")
            else:
                parts.append(f"already fits your {views}")
        if self.holes:
            parts.append(f"{self.holes} see-through hole{'s' if self.holes > 1 else ''} opened")
        return "; ".join(parts) or "no change"


@dataclass
class GuideResult:
    mesh: trimesh.Trimesh
    stats: GuideStats
    changed: bool  # False: the mesh already fitted, `mesh` is the input
    pinned: Callable[[np.ndarray], np.ndarray]  # points -> bool: surface fixed by a guide view's outline
    overlays: List[Tuple[Image.Image, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------- holes
def see_through_holes(alpha: np.ndarray, ink: np.ndarray, min_area: float = 0.002) -> Tuple[np.ndarray, np.ndarray]:
    """Holes through the object, in the sketch frame. Returns (holes, foreground) masks.

    A hole is an enclosed background region of the generated image's alpha that also matches an enclosed
    loop of the sketch's ink: most of it lies inside one loop and it fills a fair share of that loop."""
    size = ink.shape
    a = cv2.resize(np.asarray(alpha, np.uint8), (size[1], size[0]), interpolation=cv2.INTER_NEAREST) > 127
    fg = ndi.binary_closing(a, iterations=1) | a
    enclosed = ndi.binary_opening(ndi.binary_fill_holes(fg) & ~fg, iterations=2)
    lines = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    loops, _ = ndi.label(ndi.binary_fill_holes(lines) & ~lines)
    loop_area = np.bincount(loops.ravel())
    labels, n = ndi.label(enclosed)
    holes = np.zeros(size, bool)
    for i in range(1, n + 1):
        region = labels == i
        area = int(region.sum())
        if area < min_area * region.size:
            continue
        inside = np.bincount(loops[region], minlength=len(loop_area))
        inside[0] = 0
        best = int(inside.argmax())
        if best and inside[best] >= 0.6 * area and area >= 0.3 * loop_area[best]:
            holes |= region
    return holes, fg


def _bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64)  # x0, y0, x1, y1


# ---------------------------------------------------------------------------------------------- guide views
@dataclass
class _ViewMap:
    """Orthographic map between a guide sketch's pixels and the view frame.
    side: row = r0 + (y_top - y) / k, col = a - z / k (front on the left) or a + z / k (front on the right)
    top:  col = c0 + (x - x_min) / k, row = a + z / k (front at the bottom of the sheet)"""

    name: str
    keep: np.ndarray  # silhouette grown by the tolerance: what may stay
    edge: np.ndarray  # px distance to the silhouette's outline
    tol_px: float
    k: float
    r0: float
    c0: float
    a: float
    sign: float  # side view: -1 front on the left, +1 on the right

    def pixels(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.name == "side":
            return self.r0 + (self.c0 - pts[:, 1]) / self.k, self.a + self.sign * pts[:, 2] / self.k
        return self.a + pts[:, 2] / self.k, self.r0 + (pts[:, 0] - self.c0) / self.k

    def _lookup(self, img: np.ndarray, pts: np.ndarray, outside) -> np.ndarray:
        r, c = self.pixels(pts)
        ri, ci = np.floor(r).astype(np.int64), np.floor(c).astype(np.int64)
        ok = (ri >= 0) & (ri < img.shape[0]) & (ci >= 0) & (ci < img.shape[1])
        out = np.full(len(pts), outside, dtype=img.dtype)
        out[ok] = img[ri[ok], ci[ok]]
        return out

    def inside(self, pts: np.ndarray) -> np.ndarray:
        return self._lookup(self.keep, pts, False)

    def on_outline(self, pts: np.ndarray) -> np.ndarray:
        return self._lookup(self.edge, pts, np.float32(1e9)) <= self.tol_px + 2.0


def _register(name: str, silhouette: np.ndarray, occ: np.ndarray, centres: Tuple[np.ndarray, ...], front_left: bool, tolerance: float) -> Optional[_ViewMap]:
    """Scale and place a guide silhouette on the voxelised mesh using the front-facing profile."""
    box = _bbox(silhouette)
    if box is None:
        return None
    xs, ys, zs = centres
    tol_px = max(2.0, tolerance * float(max(box[2] - box[0], box[3] - box[1])))
    keep = cv2.dilate(silhouette.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(tol_px) + 1,) * 2)) > 0
    outline = silhouette & ~(cv2.erode(silhouette.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    edge = cv2.distanceTransform((~outline).astype(np.uint8), cv2.DIST_L2, 3)
    occupied_x, occupied_y = np.nonzero(occ.any(axis=(1, 2)))[0], np.nonzero(occ.any(axis=(0, 2)))[0]
    if name == "side":
        y_lo, y_hi = ys[occupied_y.min()], ys[occupied_y.max()]
        k = (y_hi - y_lo) / max(box[3] - box[1] - 1, 1)  # model units per pixel, from the height
        r0, c0, sign = box[1] + 0.5, y_hi, (-1.0 if front_left else 1.0)  # top row centre <-> top of the mesh
        offsets = []
        for j in occupied_y:
            zf = zs[np.nonzero(occ[:, j, :].any(axis=0))[0].max()]  # front-most z at this height
            row = int(np.clip(np.floor(r0 + (y_hi - ys[j]) / k), 0, silhouette.shape[0] - 1))
            cols = np.nonzero(silhouette[row])[0]
            if cols.size:
                front_col = cols.min() if front_left else cols.max()
                offsets.append(front_col + 0.5 - sign * zf / k)
    else:
        x_lo, x_hi = xs[occupied_x.min()], xs[occupied_x.max()]
        k = (x_hi - x_lo) / max(box[2] - box[0] - 1, 1)  # from the width
        r0, c0, sign = box[0] + 0.5, x_lo, 1.0  # top view: r0 / c0 are the column origin and x_min
        offsets = []
        for i in occupied_x:
            zf = zs[np.nonzero(occ[i].any(axis=0))[0].max()]
            col = int(np.clip(np.floor(r0 + (xs[i] - x_lo) / k), 0, silhouette.shape[1] - 1))
            rows = np.nonzero(silhouette[:, col])[0]
            if rows.size:
                offsets.append(rows.max() + 0.5 - zf / k)  # the front is the bottom edge of the top view
    if not offsets or k <= 0:
        return None
    return _ViewMap(name, keep, edge, tol_px, float(k), float(r0), float(c0), float(np.median(offsets)), sign)


def _overlay(view: _ViewMap, silhouette: np.ndarray, before: np.ndarray, after: np.ndarray) -> Image.Image:
    """Light blue = TripoSR's profile, dark blue = after trimming, red = the guide outline."""
    img = np.full(silhouette.shape + (3,), 255, np.uint8)
    for pts, colour in ((before, (190, 215, 255)), (after, (60, 110, 220))):
        r, c = view.pixels(pts)
        ri, ci = np.floor(r).astype(int), np.floor(c).astype(int)
        ok = (ri >= 0) & (ri < img.shape[0]) & (ci >= 0) & (ci < img.shape[1])
        m = np.zeros(silhouette.shape, np.uint8)
        m[ri[ok], ci[ok]] = 1
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        img[m] = colour
    outline = silhouette & ~(cv2.erode(silhouette.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    img[cv2.dilate(outline.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (220, 40, 40)
    box = _bbox(silhouette | (img[..., 2] < 250))
    if box is not None:
        pad = 12
        x0, y0, x1, y1 = (int(v) for v in box)
        img = img[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad]
    return Image.fromarray(img)


def apply_guides(
    mesh: trimesh.Trimesh,
    camera: CameraSpec,
    holes: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    side: Optional[np.ndarray] = None,
    top: Optional[np.ndarray] = None,
    side_front_left: bool = True,
    resolution: int = 192,
    tolerance: float = 0.03,
    smooth_iterations: int = 3,
) -> Tuple[Optional[GuideResult], str]:
    """Trim `mesh` to the side/top silhouettes and drill the see-through `holes` (holes, foreground masks in
    the sketch frame). Returns (result, reason); result is None when there is nothing to do or it failed."""
    has_holes = holes is not None and holes[0].any()
    if side is None and top is None and not has_holes:
        return None, ""
    vertices = np.asarray(mesh.vertices, np.float64)
    lo, hi = vertices.min(0), vertices.max(0)
    pitch = float((hi - lo).max()) / resolution
    pad = 2
    origin = lo - pad * pitch
    shape = tuple(np.ceil((hi - lo) / pitch).astype(int) + 2 * pad + 1)
    occ = voxelize_solid(vertices, np.asarray(mesh.faces), origin, pitch, shape)
    if occ.sum() < 1000:
        return None, "the mesh could not be voxelised as a closed solid"
    centres = tuple(origin[a] + (np.arange(shape[a]) + 0.5) * pitch for a in range(3))
    idx = np.argwhere(occ)
    pts = origin + (idx + 0.5) * pitch
    keep = np.ones(len(pts), bool)

    views: List[_ViewMap] = []
    sils = {}
    for name, sil in (("side", side), ("top", top)):
        if sil is None:
            continue
        view = _register(name, np.asarray(sil, bool), occ, centres, side_front_left, tolerance)
        if view is None:
            return None, f"the {name} view is empty"
        views.append(view)
        sils[name] = np.asarray(sil, bool)
        keep &= view.inside(pts)
    trimmed = ~keep

    drilled = np.zeros(len(pts), bool)
    n_holes = 0
    if has_holes:
        hole_mask, fg = holes
        n_holes = int(ndi.label(hole_mask)[1])
        hole_mask = ndi.binary_erosion(hole_mask, iterations=2)  # stay clear of the rim: framing is not pixel-exact
        size = 256
        render = render_silhouette(vertices, np.asarray(mesh.faces), camera, size)
        rb, ab = _bbox(render), _bbox(fg)
        if rb is not None and ab is not None:
            p = project(pts, camera, size)
            q = ab[:2] + (p + 0.5 - rb[:2]) * (ab[2:] - ab[:2]) / (rb[2:] - rb[:2])
            qi = np.floor(q).astype(np.int64)
            ok = (qi[:, 0] >= 0) & (qi[:, 0] < hole_mask.shape[1]) & (qi[:, 1] >= 0) & (qi[:, 1] < hole_mask.shape[0])
            drilled[ok] = hole_mask[qi[ok, 1], qi[ok, 0]]
        drilled &= keep

    total = len(pts)
    trimmed_fraction, hole_fraction = float(trimmed.sum() / total), float(drilled.sum() / total)
    if trimmed_fraction > 0.65:
        return None, "the guide views and the model hardly overlap (is the side view drawn facing the right way?)"
    ext = hi - lo
    depth_before = float(ext[2] / max(ext[0], ext[1]))

    def pinned(points: np.ndarray) -> np.ndarray:
        out = np.zeros(len(points), bool)
        for v in views:
            out |= v.on_outline(np.asarray(points, np.float64))
        return out

    remove = trimmed | drilled
    if remove.sum() < 0.002 * total:  # nothing worth re-meshing for
        stats = GuideStats([v.name for v in views], round(trimmed_fraction, 4), 0, 0.0, round(depth_before, 4), round(depth_before, 4), len(mesh.faces))
        overlays = [(_overlay(v, sils[v.name], pts, pts), f"{v.name} view guide: fits already") for v in views]
        return GuideResult(mesh, stats, False, pinned, overlays), ""

    new_occ = occ.copy()
    gone = idx[remove]
    new_occ[gone[:, 0], gone[:, 1], gone[:, 2]] = False
    labels, n = ndi.label(new_occ, structure=np.ones((3, 3, 3)))
    if n == 0:
        return None, "nothing of the model is left inside the guide views"
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    new_occ = (sizes >= 0.03 * sizes.max())[labels]  # drop crumbs the cut left behind

    from skimage.measure import marching_cubes

    verts, faces, _, _ = marching_cubes(ndi.gaussian_filter(np.pad(new_occ, 1).astype(np.float32), 0.8), 0.5)
    verts = origin + (verts - 1 + 0.5) * pitch
    if signed_volume(verts, faces) < 0:
        faces = faces[:, ::-1]
    colors = None
    if mesh.visual.kind == "vertex":
        _, nearest = cKDTree(vertices).query(verts)
        colors = np.asarray(mesh.visual.vertex_colors)[nearest]
    out = smooth_mesh(trimesh.Trimesh(verts, np.ascontiguousarray(faces), vertex_colors=colors, process=False), smooth_iterations)
    ext2 = out.vertices.max(0) - out.vertices.min(0)
    stats = GuideStats(
        views=[v.name for v in views],
        trimmed_fraction=round(trimmed_fraction, 4),
        holes=n_holes if drilled.any() else 0,
        hole_fraction=round(hole_fraction, 4),
        depth_before=round(depth_before, 4),
        depth_after=round(float(ext2[2] / max(ext2[0], ext2[1])), 4),
        faces=int(len(out.faces)),
    )
    kept = pts[~remove]
    overlays = [(_overlay(v, sils[v.name], pts, kept), f"{v.name} view guide: light blue = TripoSR, dark blue = after trimming, red = your outline") for v in views]
    return GuideResult(out, stats, True, pinned, overlays), ""
