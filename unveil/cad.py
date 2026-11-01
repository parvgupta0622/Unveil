"""CAD drawing -> 3D: a solid from a DXF/DWG with three orthographic views (proof of concept).

Expected drawing (what a clean AutoCAD part drawing usually looks like):
- Exactly three views of one part: front, top and side, in model space, without a border or title block
  (or with those on layers named BORDER / TITLE / DIM..., which are ignored).
- Either the views sit on layers named FRONT, TOP and SIDE (PLAN, ELEVATION, RIGHT, LEFT also work), or they
  are laid out the standard way: top view in line with the front view (above it = third-angle, below it =
  first-angle projection) and the side view beside the front view at the same height.
- Visible edges as continuous lines; hidden edges with a HIDDEN / DASHED linetype (or on a HIDDEN layer);
  centre lines (CENTER / DASHDOT linetype or layer) are ignored. Text and dimensions are ignored.

How the solid is built (a "visual hull" from the three outlines):
1. Each view's visible lines are rasterised and filled from the outside: that view's silhouette.
2. The solid is every point whose projections fall inside all three silhouettes (front x top x side).
3. Holes: a closed shape inside a view (e.g. a circle in the top view) is cut out only when another view
   shows it with hidden lines along the cutting direction; the hidden lines' extent sets the hole depth, so
   through holes and blind holes both work. A closed shape without hidden-line evidence stays solid (a boss).
4. Marching cubes on the voxel grid; units stay those of the drawing (usually mm).

Limits: shapes a three-silhouette intersection cannot express are approximated (a sphere comes out as the
intersection of three cylinders, a cone viewed from the side gets flat facets), inclined views and
sections are not supported.

DWG is a closed binary format: it is converted to DXF first with LibreDWG's `dwg2dxf` (built by the Colab
notebook) or, when installed, the ODA File Converter. DXF files are read directly with ezdxf.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from .mesh.processing import signed_volume
from .utils import LOG

VIEW_NAMES = ("front", "top", "side")
# Layer-name keywords for each view (matched as words inside the layer name, case-insensitive).
VIEW_LAYER_WORDS = {
    "front": ("FRONT", "ELEVATION", "ELEV"),
    "top": ("TOP", "PLAN"),
    "side": ("SIDE", "RIGHT", "LEFT", "PROFILE", "END"),
}
IGNORED_LAYER_WORDS = ("DIM", "TEXT", "TITLE", "BORDER", "FRAME", "NOTE", "DEFPOINTS", "HATCH", "VIEWPORT")
# AutoCAD names plus the ISO 128 ones (ACAD_ISO02W100 = dashed, ISO04/08/10 = long-dash-dot centre lines).
HIDDEN_WORDS = ("HIDDEN", "DASHED", "DASH", "ISO02W100", "ISO03W100")
CENTER_WORDS = ("CENTER", "CENTRE", "DASHDOT", "PHANTOM", "CL", "ISO04W100", "ISO08W100", "ISO10W100")
CURVE_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"}
MODEL_COLOR = (205, 214, 218, 255)
# Axes of each view in the part's frame (x right, y back = away from the front viewer, z up):
# (horizontal axis, vertical axis, viewing / cutting axis).
VIEW_AXES = {"front": (0, 2, 1), "top": (0, 1, 2), "side": (1, 2, 0)}
AXIS_NAMES = "xyz"


class CadError(ValueError):
    """The drawing cannot be turned into a model (message is shown to the user)."""


@dataclass
class Curve:
    points: np.ndarray  # (N, 2) sheet coordinates
    closed: bool
    hidden: bool
    layer: str
    kind: str

    @property
    def bbox(self) -> np.ndarray:
        return np.r_[self.points.min(0), self.points.max(0)]


@dataclass
class View:
    name: str
    curves: List[Curve]
    bbox: np.ndarray  # sheet xmin, ymin, xmax, ymax
    flip_h: bool = False  # side view: sheet rightwards = towards the front (first-angle left view, ...)
    title: str = ""

    @property
    def size(self) -> np.ndarray:
        return self.bbox[2:] - self.bbox[:2]

    def to_part(self, pts: np.ndarray) -> np.ndarray:
        """Sheet points -> (horizontal, vertical) coordinates on this view's part axes, starting at 0.
        Top view: up = towards the back in both projection methods. Side view: see `flip_h`."""
        uv = pts - self.bbox[:2]
        if self.flip_h:
            uv = np.c_[self.size[0] - uv[:, 0], uv[:, 1]]
        return uv


@dataclass
class Hole:
    view: str
    mask: np.ndarray  # pixels of the view grid
    depth_range: Tuple[float, float]  # along the view's cutting axis, part units
    evidence: str  # which view's hidden lines confirmed it


@dataclass
class CadModel:
    mesh: trimesh.Trimesh  # view frame (+x right, +y up, +z towards the front viewer), drawing units
    dimensions: Tuple[float, float, float]  # width (x), depth, height in drawing units
    projection: str  # "third-angle" | "first-angle"
    view_source: str  # "layers" | "layout"
    holes: List[Hole]
    preview: Image.Image
    silhouettes: Dict[str, Image.Image]
    units: str
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        w, d, h = self.dimensions
        return {
            "dimensions": {"width": round(w, 4), "depth": round(d, 4), "height": round(h, 4), "units": self.units},
            "projection": self.projection,
            "views_from": self.view_source,
            "holes": [{"view": h.view, "depth_range": [round(v, 4) for v in h.depth_range], "evidence": h.evidence} for h in self.holes],
            "faces": int(len(self.mesh.faces)),
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------------------------- reading
def _dwg_converter() -> Optional[str]:
    """Name of an available DWG -> DXF converter."""
    if shutil.which("dwg2dxf"):
        return "dwg2dxf"
    try:
        from ezdxf.addons import odafc

        if odafc.is_installed():
            return "odafc"
    except Exception:  # noqa: BLE001 - odafc probing is best effort
        pass
    return None


def convert_dwg(path: Path, workdir: Path) -> Path:
    tool = _dwg_converter()
    out = Path(workdir) / (Path(path).stem + ".dxf")
    if tool == "dwg2dxf":
        proc = subprocess.run(["dwg2dxf", "-y", "-o", str(out), str(path)], capture_output=True, text=True, timeout=300)
        if not out.is_file():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            raise CadError("LibreDWG could not convert this DWG: " + (" ".join(tail) or f"exit code {proc.returncode}") + ". Save it as DXF in AutoCAD and upload that instead.")
        return out
    if tool == "odafc":
        from ezdxf.addons import odafc

        odafc.convert(str(path), str(out), version="R2018", replace=True)
        return out
    raise CadError(
        "Reading .dwg needs a DWG converter, and none is installed. In Colab run the notebook's 'DWG support' "
        "cell (builds LibreDWG), or save the drawing as DXF in AutoCAD (Save As → DXF) and upload that."
    )


def read_drawing(path: Path):
    """ezdxf document for a .dxf or .dwg file."""
    import ezdxf
    from ezdxf import recover

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".dxf", ".dwg"):
        raise CadError(f"Upload a .dxf or .dwg file (got '{path.suffix or path.name}').")
    with tempfile.TemporaryDirectory() as tmp:
        source = convert_dwg(path, Path(tmp)) if suffix == ".dwg" else path
        try:
            doc, auditor = recover.readfile(str(source))
        except (OSError, ezdxf.DXFStructureError) as exc:
            raise CadError(f"Could not read the drawing: {exc}") from exc
    if auditor.has_errors:
        LOG.warning("DXF had %d structural errors (recovered)", len(auditor.errors))
    return doc


def _words(name: str) -> List[str]:
    out, cur = [], ""
    for ch in name.upper():
        if ch.isalnum():
            cur += ch
        else:
            out.append(cur)
            cur = ""
    return [w for w in out + [cur] if w]


def _has_word(name: str, words: Sequence[str]) -> bool:
    """Whole-word match, or a word that starts with the keyword ("TOPVIEW", "HIDDEN2"); 2-letter keys exact."""
    ws = _words(name)
    return any(w in ws or (len(w) >= 3 and any(x.startswith(w) for x in ws)) for w in words)


def _linetype(entity, doc) -> str:
    lt = (entity.dxf.get("linetype") or "BYLAYER").upper()
    if lt == "BYLAYER":
        name = entity.dxf.get("layer", "0")
        layer = doc.layers.get(name) if doc.layers.has_entry(name) else None
        lt = (layer.dxf.get("linetype") or "CONTINUOUS").upper() if layer is not None else "CONTINUOUS"
    return lt


def _entities(container, depth: int = 0):
    for e in container:
        if e.dxftype() == "INSERT" and depth < 4:
            try:
                yield from _entities(e.virtual_entities(), depth + 1)
            except Exception:  # noqa: BLE001 - a broken block reference is skipped, not fatal
                continue
        else:
            yield e


def extract_curves(doc) -> List[Curve]:
    """Visible and hidden 2D curves of the model space, flattened to polylines."""
    from ezdxf import bbox as ezbbox
    from ezdxf import path as ezpath

    msp = doc.modelspace()
    ext = ezbbox.extents(msp)
    if not ext.has_data:
        raise CadError("The drawing's model space is empty.")
    size = float(max(ext.size.x, ext.size.y, 1e-9))
    tol = size / 3000.0
    curves = []
    for e in _entities(msp):
        kind = e.dxftype()
        if kind not in CURVE_TYPES:
            continue
        layer = e.dxf.get("layer", "0")
        lt = _linetype(e, doc)
        if _has_word(layer, IGNORED_LAYER_WORDS) or _has_word(lt, CENTER_WORDS) or _has_word(layer, CENTER_WORDS):
            continue
        try:
            p = ezpath.make_path(e)
            pts = np.array([(v.x, v.y) for v in p.flattening(distance=tol)], dtype=np.float64)
        except Exception:  # noqa: BLE001 - an entity ezdxf cannot flatten is skipped
            continue
        if len(pts) < 2 or not np.isfinite(pts).all():
            continue
        closed = kind == "CIRCLE" or bool(np.linalg.norm(pts[0] - pts[-1]) <= tol)
        hidden = _has_word(lt, HIDDEN_WORDS) or _has_word(layer, ("HIDDEN",))
        curves.append(Curve(pts, closed, hidden, layer, kind))
    if not any(not c.hidden for c in curves):
        raise CadError("No visible lines found (only text, dimensions or hidden lines).")
    return curves


# ---------------------------------------------------------------------------------------------- views
def _clusters(curves: List[Curve], gap: float) -> List[List[int]]:
    """Groups of curves whose bounding boxes touch (within `gap`): one group per drawn view."""
    boxes = np.array([c.bbox for c in curves])
    parent = list(range(len(curves)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lo, hi = boxes[:, :2] - gap, boxes[:, 2:] + gap
    for i in range(len(curves)):
        touch = np.flatnonzero((lo[i] <= boxes[:, 2:]).all(1) & (hi[i] >= boxes[:, :2]).all(1))
        for j in touch:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[ri] = rj
    groups: Dict[int, List[int]] = {}
    for i in range(len(curves)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _bbox(curves: Sequence[Curve]) -> np.ndarray:
    b = np.array([c.bbox for c in curves])
    return np.r_[b[:, :2].min(0), b[:, 2:].max(0)]


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0)) / max(min(a1 - a0, b1 - b0), 1e-12)


def _views_from_layers(curves: List[Curve]) -> Optional[Dict[str, List[Curve]]]:
    named: Dict[str, List[Curve]] = {v: [] for v in VIEW_NAMES}
    others = []
    for c in curves:
        hits = [v for v in VIEW_NAMES if _has_word(c.layer, VIEW_LAYER_WORDS[v])]
        (named[hits[0]] if len(hits) == 1 else others).append(c)
    if not all(any(not c.hidden for c in named[v]) for v in VIEW_NAMES):
        return None
    boxes = {v: _bbox(named[v]) for v in VIEW_NAMES}
    for c in others:  # hidden lines on a HIDDEN layer etc.: the view whose box contains / is nearest them
        centre = 0.5 * (c.bbox[:2] + c.bbox[2:])
        dist = {v: np.linalg.norm(np.maximum(0, np.maximum(b[:2] - centre, centre - b[2:]))) for v, b in boxes.items()}
        named[min(dist, key=dist.get)].append(c)
    return named


def _views_from_layout(curves: List[Curve], size: float) -> Tuple[Dict[str, List[Curve]], List[str]]:
    groups = _clusters(curves, gap=0.01 * size)
    groups = [g for g in groups if any(not curves[i].hidden for i in g)]
    boxes = [_bbox([curves[i] for i in g]) for g in groups]
    area = [float(np.prod(np.maximum(b[2:] - b[:2], size * 1e-3))) for b in boxes]
    order = np.argsort(area)[::-1]
    if len(groups) < 3:
        raise CadError(
            f"Found {len(groups)} separate drawing(s); expected three views (front, top, side) with space between them. "
            "Remove any border/title block or put the views on layers named FRONT, TOP and SIDE."
        )
    warnings = []
    big = [int(i) for i in order[:3]]
    if len(groups) > 3:
        warnings.append(f"Ignored {len(groups) - 3} small separate item(s) outside the three main views.")
    b = {i: boxes[i] for i in big}
    col = {(i, j) for i in big for j in big if i != j and _overlap(b[i][0], b[i][2], b[j][0], b[j][2]) > 0.6}
    row = {(i, j) for i in big for j in big if i != j and _overlap(b[i][1], b[i][3], b[j][1], b[j][3]) > 0.6}
    for f in big:
        tops = [j for (i, j) in col if i == f]
        sides = [j for (i, j) in row if i == f]
        if len(tops) == 1 and len(sides) == 1 and tops[0] != sides[0]:
            pick = {"front": f, "top": tops[0], "side": sides[0]}
            return {v: [curves[k] for k in groups[pick[v]]] for v in VIEW_NAMES}, warnings
    raise CadError(
        "Could not tell which view is which. Lay them out the standard way (top view directly above or below the "
        "front view, side view level with it) or put them on layers named FRONT, TOP and SIDE."
    )


def find_views(curves: List[Curve]) -> Tuple[Dict[str, View], str, str, List[str]]:
    """(views, projection, source, warnings)."""
    all_box = _bbox(curves)
    size = float(max(all_box[2:] - all_box[:2]))
    warnings: List[str] = []
    grouped = _views_from_layers(curves)
    source = "layers"
    if grouped is None:
        grouped, warnings = _views_from_layout(curves, size)
        source = "layout"
    views = {v: View(v, grouped[v], _bbox([c for c in grouped[v] if not c.hidden])) for v in VIEW_NAMES}
    front, top, side = views["front"].bbox, views["top"].bbox, views["side"].bbox
    top_above = (top[1] + top[3]) > (front[1] + front[3])
    side_right = (side[0] + side[2]) > (front[0] + front[2])
    # Third-angle: top above, right view on the right. First-angle: plan below, left view on the right.
    # Either way the side view's sheet-rightwards direction points to the back iff top_above == side_right.
    views["side"].flip_h = top_above != side_right
    projection = "third-angle" if top_above else "first-angle"
    # Third-angle draws each view on the side it is seen from; first-angle on the opposite side.
    seen_from_right = side_right == top_above
    for name, view in views.items():
        view.title = {"front": "FRONT", "top": "TOP", "side": "RIGHT SIDE" if seen_from_right else "LEFT SIDE"}[name]

    fs, ts, ss = views["front"].size, views["top"].size, views["side"].size
    for what, a, b in (("width (front vs top)", fs[0], ts[0]), ("height (front vs side)", fs[1], ss[1]), ("depth (top vs side)", ts[1], ss[0])):
        if abs(a - b) > 0.03 * max(a, b):
            warnings.append(f"The views disagree on the {what}: {a:.4g} vs {b:.4g}; the larger is used.")
    return views, projection, source, warnings


# ---------------------------------------------------------------------------------------------- rasters
SUPER = 4  # supersampling of the line rasters


def _raster_lines(view: View, curves: Sequence[Curve], shape: Tuple[int, int], pitch: float, thickness: int = 1) -> np.ndarray:
    """Lines on a (h, v) canvas at SUPER x resolution, padded by 2 pixels; index [h, v]."""
    nh, nv = shape
    canvas = np.zeros(((nv + 4) * SUPER, (nh + 4) * SUPER), np.uint8)  # rows = v
    shift = 4
    for c in curves:
        uv = view.to_part(c.points) / pitch * SUPER + 2 * SUPER
        pts = np.round(uv * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], bool(c.closed), 255, thickness, cv2.LINE_8, shift)
    return canvas.T > 0  # [h, v]


def _down(mask_hi: np.ndarray) -> np.ndarray:
    """SUPER x supersampled padded mask -> pixel grid (area > half)."""
    h, v = mask_hi.shape[0] // SUPER, mask_hi.shape[1] // SUPER
    m = mask_hi[: h * SUPER, : v * SUPER].reshape(h, SUPER, v, SUPER).mean(axis=(1, 3)) > 0.5
    return m[2:-2, 2:-2]


def view_regions(view: View, shape: Tuple[int, int], pitch: float):
    """Silhouette (filled outline) of a view plus the candidate hole regions inside it, on the pixel grid."""
    visible = [c for c in view.curves if not c.hidden]
    for close in (1, 3, 7):  # small gaps between line ends would let the outside flood in: close them
        lines = _raster_lines(view, visible, shape, pitch)
        if close > 1:
            lines = ndi.binary_closing(lines, structure=np.ones((close, close)), border_value=0) | lines
        free = ~lines
        labels, _ = ndi.label(free)
        outside = labels == labels[0, 0]
        solid_hi = ~outside
        if _down(solid_hi).sum() > 0.02 * shape[0] * shape[1]:
            break
    else:
        raise CadError(f"The {view.name} view's outline is not closed, so its inside cannot be told from its outside.")

    # Hole candidates: regions enclosed by a single loop of lines that does not touch the outline.
    line_lab, _ = ndi.label(lines, structure=np.ones((3, 3)))
    outline_ids = set(np.unique(line_lab[ndi.binary_dilation(outside) & lines])) - {0}
    region_lab, n_regions = ndi.label(free & ~outside)
    holes = []
    for r in range(1, n_regions + 1):
        region = region_lab == r
        ring = set(np.unique(line_lab[ndi.binary_dilation(region, iterations=2) & lines])) - {0}
        if len(ring) == 1 and not ring & outline_ids:
            mask = _down(region | (ndi.binary_dilation(region, iterations=2) & lines))
            if mask.any():
                holes.append(mask)
    return _down(solid_hi), holes


def _hole_depth(view: View, mask: np.ndarray, views: Dict[str, View], pitch: float, dims: np.ndarray) -> Optional[Tuple[float, float, str]]:
    """Hidden-line evidence for a hole: another view showing hidden lines along the cutting axis at both edges
    of the hole. Returns (depth from, depth to, evidence view) or None."""
    h_ax, v_ax, cut_ax = VIEW_AXES[view.name]
    idx = np.argwhere(mask)
    extent = {h_ax: (idx[:, 0].min() * pitch, (idx[:, 0].max() + 1) * pitch), v_ax: (idx[:, 1].min() * pitch, (idx[:, 1].max() + 1) * pitch)}
    tol = max(1.5 * pitch, 0.02 * float(dims.max()))
    for other in views.values():
        if other.name == view.name:
            continue
        oh, ov, _ = VIEW_AXES[other.name]
        if cut_ax not in (oh, ov):
            continue
        shared = oh if ov == cut_ax else ov
        s_col, d_col = (0, 1) if oh == shared else (1, 0)
        found = {0: [], 1: []}
        for c in other.curves:
            if not c.hidden:
                continue
            p = other.to_part(c.points)
            s, d = p[:, s_col], p[:, d_col]
            if np.ptp(s) > tol or np.ptp(d) < tol:
                continue  # not a line along the cutting axis
            for k, edge in enumerate(extent[shared]):
                if abs(float(np.mean(s)) - edge) <= tol:
                    found[k].append((float(d.min()), float(d.max())))
        if found[0] and found[1]:
            spans = found[0] + found[1]
            lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
            if hi - lo >= 0.95 * dims[cut_ax]:
                lo, hi = 0.0, float(dims[cut_ax])
            return lo, hi, other.name
    return None


# ---------------------------------------------------------------------------------------------- solid
def build_solid(views: Dict[str, View], resolution: int = 192) -> Tuple[trimesh.Trimesh, List[Hole], Dict[str, np.ndarray], np.ndarray]:
    fs, ts, ss = views["front"].size, views["top"].size, views["side"].size
    dims = np.array([max(fs[0], ts[0]), max(ts[1], ss[0]), max(fs[1], ss[1])])  # x width, y depth, z height
    if (dims <= 0).any():
        raise CadError("One of the views is flat (zero width or height).")
    pitch = float(dims.max()) / resolution
    n = np.maximum(np.ceil(dims / pitch).astype(int), 1)
    sil, holes = {}, []
    for name, view in views.items():
        h_ax, v_ax, _ = VIEW_AXES[name]
        sil[name], candidates = view_regions(view, (int(n[h_ax]), int(n[v_ax])), pitch)
        for mask in candidates:
            hit = _hole_depth(view, mask, views, pitch, dims)
            if hit is not None:
                holes.append(Hole(name, mask, (hit[0], hit[1]), hit[2]))

    occ = sil["front"][:, None, :] & sil["top"][:, :, None] & sil["side"][None, :, :]  # [x, y, z]
    for hole in holes:
        h_ax, v_ax, cut_ax = VIEW_AXES[hole.view]
        k0 = int(np.floor(hole.depth_range[0] / pitch))
        k1 = int(np.ceil(hole.depth_range[1] / pitch))
        cut = np.zeros(n[cut_ax], bool)
        cut[max(k0, 0) : min(k1, n[cut_ax])] = True
        shape = [1, 1, 1]
        shape[h_ax], shape[v_ax], shape[cut_ax] = int(n[h_ax]), int(n[v_ax]), int(n[cut_ax])
        prism = np.moveaxis(hole.mask[:, :, None] & cut[None, None, :], [0, 1, 2], [h_ax, v_ax, cut_ax])
        occ &= ~prism
    if occ.sum() < 50:
        raise CadError("The three views do not overlap into a solid; check that they are drawn to the same scale.")

    from skimage.measure import marching_cubes

    field_ = ndi.gaussian_filter(np.pad(occ, 2).astype(np.float32), 0.6)
    verts, faces, _, _ = marching_cubes(field_, 0.5)
    part = (verts - 2 + 0.5) * pitch  # voxel centres sit at (i + 0.5) * pitch
    view_frame = np.c_[part[:, 0], part[:, 2], -part[:, 1]]  # x right, z up, front (y = 0) towards the camera
    if signed_volume(view_frame, faces) < 0:
        faces = faces[:, ::-1]
    colors = np.tile(np.array(MODEL_COLOR, np.uint8), (len(view_frame), 1))
    mesh = trimesh.Trimesh(view_frame, np.ascontiguousarray(faces), vertex_colors=colors, process=False)
    return mesh, holes, sil, dims


# ---------------------------------------------------------------------------------------------- previews
VIEW_COLORS = {"front": (26, 83, 92), "top": (31, 140, 132), "side": (214, 120, 30)}


def _preview(views: Dict[str, View], projection: str, holes: List[Hole], size: int = 900) -> Image.Image:
    curves = [c for v in views.values() for c in v.curves]
    box = _bbox(curves)
    span = float(max(box[2:] - box[:2]))
    scale = (size - 80) / span
    w = int((box[2] - box[0]) * scale) + 80
    h = int((box[3] - box[1]) * scale) + 80
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()

    def sheet(p):
        return [(40 + (x - box[0]) * scale, h - 40 - (y - box[1]) * scale) for x, y in p]

    for name, view in views.items():
        col = VIEW_COLORS[name]
        for c in view.curves:
            pts = sheet(c.points)
            if c.hidden:
                light = tuple(int(0.45 * a + 0.55 * 255) for a in col)
                for seg in _dashes(np.asarray(pts), dash=12.0, gap=7.0):
                    draw.line(seg, fill=light, width=2)
            else:
                draw.line(pts + ([pts[0]] if c.closed else []), fill=col, width=3)
        x0, y0 = sheet([view.bbox[:2]])[0]
        n_holes = sum(hh.view == name for hh in holes)
        label = view.title + (f" · {n_holes} hole(s)" if n_holes else "")
        draw.text((x0, y0 + 6), label, fill=col, font=font)
    draw.text((10, 10), f"{projection} projection · hidden lines dashed", fill=(80, 80, 80), font=font)
    return img


def _dashes(pts: np.ndarray, dash: float, gap: float) -> List[List[Tuple[float, float]]]:
    """Split a polyline (pixel coordinates) into dash segments."""
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg_len)]
    out, start = [], 0.0
    while start < cum[-1]:
        stops = np.linspace(start, min(start + dash, cum[-1]), 6)
        out.append([tuple(map(float, (np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])))) for t in stops])
        start += dash + gap
    return out


def _mask_image(mask: np.ndarray, holes: Sequence[np.ndarray] = ()) -> Image.Image:
    img = np.full(mask.T.shape + (3,), 248, np.uint8)
    m = np.flipud(mask.T)
    img[m] = (26, 83, 92)
    for hm in holes:
        img[np.flipud(hm.T)] = (255, 107, 107)
    return Image.fromarray(img).resize((max(64, img.shape[1] * 2), max(64, img.shape[0] * 2)), Image.NEAREST)


# ---------------------------------------------------------------------------------------------- entry point
def cad_to_model(path: Path, resolution: int = 192) -> CadModel:
    doc = read_drawing(path)
    curves = extract_curves(doc)
    views, projection, source, warnings = find_views(curves)
    mesh, holes, sil, dims = build_solid(views, resolution)
    return assemble(doc, curves, views, projection, source, mesh, holes, sil, dims, warnings)


def assemble(doc, curves, views, projection, source, mesh, holes, sil, dims, warnings) -> CadModel:
    silhouettes = {name: _mask_image(sil[name], [h.mask for h in holes if h.view == name]) for name in VIEW_NAMES}
    if not any(c.hidden for c in curves):
        warnings.append("No hidden (dashed) lines in the drawing, so closed shapes inside a view were kept solid.")
    return CadModel(mesh, (float(dims[0]), float(dims[1]), float(dims[2])), projection, source, holes, _preview(views, projection, holes), silhouettes, _units(doc), warnings)


def _units(doc) -> str:
    try:
        from ezdxf import units

        code = int(doc.header.get("$INSUNITS", 0))
        return units.unit_name(code) if code else "drawing units"
    except Exception:  # noqa: BLE001
        return "drawing units"


def converter_status() -> str:
    tool = _dwg_converter()
    return {"dwg2dxf": "LibreDWG", "odafc": "ODA File Converter"}.get(tool or "", "") or os.environ.get("UNVEIL_DWG_HINT", "")
