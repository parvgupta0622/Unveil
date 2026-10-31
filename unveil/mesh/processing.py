"""Mesh cleanup: floater removal, winding repair, smoothing, decimation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ..utils import LOG


class MeshError(RuntimeError):
    pass


@dataclass
class CleanStats:
    faces_in: int
    faces_out: int
    components: int
    retention: float  # fraction of faces in the largest component (1.0 = no floaters)
    flipped: bool


def _subset(mesh: trimesh.Trimesh, face_mask: np.ndarray) -> trimesh.Trimesh:
    """Keep the selected faces and only the vertices they use, carrying vertex colours along."""
    faces = mesh.faces[face_mask]
    used, inverse = np.unique(faces.ravel(), return_inverse=True)
    colors = None
    if mesh.visual.kind == "vertex":
        colors = np.asarray(mesh.visual.vertex_colors)[used]
    return trimesh.Trimesh(mesh.vertices[used], inverse.reshape(-1, 3), vertex_colors=colors, process=False)


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces].astype(np.float64)
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def clean_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, CleanStats]:
    """Drop invalid faces, keep only the largest connected component, orient faces outward."""
    faces_in = len(mesh.faces)
    if faces_in == 0:
        raise MeshError("mesh has no faces")
    mesh = mesh.copy()
    mesh.merge_vertices()

    v, f = mesh.vertices, mesh.faces
    finite = np.isfinite(v).all(axis=1)
    valid = finite[f].all(axis=1) & (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    tri = v[f]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid &= np.nan_to_num(area2) > 1e-12
    if not valid.any():
        raise MeshError("mesh has no valid faces")
    mesh = _subset(mesh, valid)

    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    counts = np.bincount(labels)
    largest = int(np.argmax(counts))
    retention = float(counts[largest] / len(mesh.faces))
    mesh = _subset(mesh, labels == largest)

    flipped = signed_volume(mesh.vertices, mesh.faces) < 0
    if flipped:  # marching-cubes backends disagree on winding; make normals point outward
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[:, ::-1], vertex_colors=mesh.visual.vertex_colors if mesh.visual.kind == "vertex" else None, process=False)

    stats = CleanStats(faces_in, len(mesh.faces), int(len(counts)), round(retention, 4), bool(flipped))
    return mesh, stats


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int) -> trimesh.Trimesh:
    """Taubin smoothing removes marching-cubes stair-stepping without shrinking the shape."""
    if iterations <= 0:
        return mesh
    mesh = mesh.copy()
    trimesh.smoothing.filter_taubin(mesh, iterations=int(iterations))
    return mesh


def decimate(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """Quadric decimation to at most `max_faces`; vertex colours are transferred by nearest neighbour."""
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        import fast_simplification
    except ImportError:
        LOG.warning("fast_simplification not installed; skipping decimation (%d faces)", len(mesh.faces))
        return mesh
    from scipy.spatial import cKDTree

    reduction = 1.0 - max_faces / len(mesh.faces)
    verts, faces = fast_simplification.simplify(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32), target_reduction=reduction)
    colors = None
    if mesh.visual.kind == "vertex":
        _, idx = cKDTree(mesh.vertices).query(verts)
        colors = np.asarray(mesh.visual.vertex_colors)[idx]
    return trimesh.Trimesh(verts, faces, vertex_colors=colors, process=False)


def depth_ratio(vertices: np.ndarray) -> float:
    ext = vertices.max(axis=0) - vertices.min(axis=0)
    return float(ext[2] / max(ext[0], ext[1], 1e-9))
