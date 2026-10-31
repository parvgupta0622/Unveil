"""Candidate verification: reproject a mesh into the input view and compare it with the sketch."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import trimesh
from PIL import Image

from ..config import VerifyConfig
from ..reconstruction.base import CameraSpec
from ..scoring import mask_iou
from .processing import CleanStats, depth_ratio


@dataclass
class CandidateScore:
    iou: float
    edge: float
    retention: float
    depth_ratio: float
    faces: int
    valid: bool
    reason: str
    total: float


def project(vertices: np.ndarray, camera: CameraSpec, size: int) -> np.ndarray:
    """Perspective-project view-frame vertices to pixel coordinates (x right, y down)."""
    depth = np.maximum(camera.distance - vertices[:, 2], 1e-3)
    focal = 1.0 / np.tan(np.radians(camera.fov_deg) / 2.0)
    x = vertices[:, 0] * focal / depth
    y = vertices[:, 1] * focal / depth
    return np.stack([(x + 1.0) * 0.5 * size - 0.5, (1.0 - y) * 0.5 * size - 0.5], axis=-1)


def render_silhouette(vertices: np.ndarray, faces: np.ndarray, camera: CameraSpec, size: int = 256) -> np.ndarray:
    pts = project(np.asarray(vertices, np.float64), camera, size)
    tris = np.round(pts[np.asarray(faces)] * 16).astype(np.int32)  # 4 bits of sub-pixel precision
    canvas = np.zeros((size, size), np.uint8)
    for tri in tris:  # fillConvexPoly per triangle: fillPoly's even-odd rule would punch holes at overlaps
        cv2.fillConvexPoly(canvas, tri, 255, cv2.LINE_8, 4)
    return canvas > 0


def normalize_mask(mask: np.ndarray, size: int = 256, margin: float = 0.1) -> np.ndarray:
    """Crop to the bounding box and fit (aspect preserved) into a centred square canvas."""
    ys, xs = np.nonzero(mask)
    out = np.zeros((size, size), bool)
    if ys.size == 0:
        return out
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8) * 255
    scale = size * (1 - 2 * margin) / max(crop.shape)
    h, w = max(1, round(crop.shape[0] * scale)), max(1, round(crop.shape[1] * scale))
    resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR) > 127
    oy, ox = (size - h) // 2, (size - w) // 2
    out[oy : oy + h, ox : ox + w] = resized
    return out


def outline(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def chamfer_similarity(edge_a: np.ndarray, edge_b: np.ndarray, sigma_px: float) -> float:
    """exp(-symmetric mean chamfer distance / sigma)."""
    if not edge_a.any() or not edge_b.any():
        return 0.0
    dist_to_b = cv2.distanceTransform((~edge_b).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_a = cv2.distanceTransform((~edge_a).astype(np.uint8), cv2.DIST_L2, 3)
    chamfer = 0.5 * (dist_to_b[edge_a].mean() + dist_to_a[edge_b].mean())
    return float(np.exp(-chamfer / sigma_px))


def overlay_image(render: np.ndarray, sketch: np.ndarray) -> Image.Image:
    """Blue fill = reprojected mesh, red line = sketch silhouette outline."""
    img = np.full(render.shape + (3,), 255, np.uint8)
    img[render] = (170, 200, 255)
    img[outline(render)] = (40, 90, 220)
    img[cv2.dilate(outline(sketch).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (220, 40, 40)
    return Image.fromarray(img)


def score_candidate(mesh: trimesh.Trimesh, sketch_silhouette: np.ndarray, camera: CameraSpec, stats: CleanStats, cfg: VerifyConfig) -> tuple[CandidateScore, Image.Image]:
    n = cfg.render_size
    render = render_silhouette(mesh.vertices, mesh.faces, camera, n)
    a = normalize_mask(render, n)
    b = normalize_mask(sketch_silhouette, n)
    iou = mask_iou(a, b)
    edge = chamfer_similarity(outline(a), outline(b), cfg.edge_sigma * n)
    dr = depth_ratio(np.asarray(mesh.vertices))

    reason = ""
    if len(mesh.faces) < cfg.min_faces:
        reason = f"too few faces ({len(mesh.faces)})"
    elif render.sum() < 20:
        reason = "mesh not visible from the input view"
    elif dr < cfg.min_depth_ratio:
        reason = f"flat geometry (depth ratio {dr:.3f})"
    total = cfg.w_iou * iou + cfg.w_edge * edge + cfg.w_retention * stats.retention
    score = CandidateScore(round(iou, 4), round(edge, 4), stats.retention, round(dr, 4), len(mesh.faces), not reason, reason, round(total, 4))
    return score, overlay_image(a, b)
