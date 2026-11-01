"""GLB / OBJ export and post-export validation (re-open the files and check what survived)."""
from __future__ import annotations

import inspect
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from .asset import MeshAsset


@dataclass
class ExportBundle:
    glb: Path
    obj: Path
    obj_zip: Path
    depth_scale: float
    textured: bool
    report: dict = field(default_factory=dict)
    heatmap: Optional[Path] = None  # confidence heatmap OBJ (viewer only, not in the downloads)


def _export_obj(mesh: trimesh.Trimesh, obj_path: Path) -> List[Path]:
    from trimesh.exchange.obj import export_obj

    params = inspect.signature(export_obj).parameters
    kwargs = {"include_normals": True, "include_color": True, "include_texture": True}
    if "return_texture" in params:
        kwargs["return_texture"] = True
    if "mtl_name" in params:
        kwargs["mtl_name"] = obj_path.stem + ".mtl"
    result = export_obj(mesh, **kwargs)
    text, extra = result if isinstance(result, tuple) else (result, {})
    obj_path.write_text(text)
    written = [obj_path]
    for name, data in (extra or {}).items():
        target = obj_path.parent / name
        target.write_bytes(data if isinstance(data, bytes) else str(data).encode())
        written.append(target)
    return written


def export_asset(asset: MeshAsset, out_dir: Path, depth_scale: float = 1.0, stem: str = "unveil") -> ExportBundle:
    """Apply depth scaling, centre, and write GLB + OBJ (+MTL/PNG) + a zip of the OBJ set."""
    out_dir = Path(out_dir)
    tag = f"{stem}_depth{depth_scale:.2f}"
    final = asset.with_depth_scale(depth_scale).centered()

    glb_path = out_dir / f"{tag}.glb"
    final.to_trimesh("pbr").export(glb_path)

    obj_dir = out_dir / f"{tag}_obj"  # own folder: trimesh's default MTL/PNG names would collide
    obj_dir.mkdir(parents=True, exist_ok=True)
    obj_path = obj_dir / f"{tag}.obj"
    files = _export_obj(final.to_trimesh("simple"), obj_path)
    zip_path = out_dir / f"{tag}_obj.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)

    bundle = ExportBundle(glb_path, obj_path, zip_path, float(depth_scale), final.textured)
    bundle.report = validate_export(bundle, final)
    return bundle


def zip_bundles(bundles: List[ExportBundle], zip_path: Path) -> Path:
    """One archive with every model in every format: <stem>.glb and <stem>_obj/<stem>.obj (+ .mtl / texture)."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for b in bundles:
            zf.write(b.glb, arcname=b.glb.name)
            for f in sorted(b.obj.parent.iterdir()):
                zf.write(f, arcname=f"{b.obj.parent.name}/{f.name}")
    return zip_path


def _has_texture(mesh: trimesh.Trimesh) -> bool:
    if not isinstance(mesh.visual, trimesh.visual.TextureVisuals) or mesh.visual.uv is None:
        return False
    mat = mesh.visual.material
    return getattr(mat, "baseColorTexture", None) is not None or getattr(mat, "image", None) is not None


def _has_vertex_colors(mesh: trimesh.Trimesh) -> bool:
    if mesh.visual.kind != "vertex":
        return False
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
    return bool(colors.std() > 1.0)  # not just trimesh's flat default grey


def validate_export(bundle: ExportBundle, asset: MeshAsset) -> dict:
    checks: List[Tuple[str, bool, str]] = []
    stats = {}
    for label, path in (("GLB", bundle.glb), ("OBJ", bundle.obj)):
        try:
            mesh = trimesh.load(path, force="mesh", process=False)
        except Exception as exc:  # noqa: BLE001 - report any loader failure as a failed check
            checks.append((f"{label} opens", False, str(exc)))
            continue
        checks.append((f"{label} opens", True, f"{path.stat().st_size / 1e6:.2f} MB"))
        nf = len(mesh.faces)
        checks.append((f"{label} non-empty geometry", nf > 0 and len(mesh.vertices) > 0, f"{len(mesh.vertices):,} verts / {nf:,} faces"))
        checks.append((f"{label} face count preserved", nf == len(asset.faces), f"{nf:,} vs {len(asset.faces):,}"))
        checks.append((f"{label} finite coordinates", bool(np.isfinite(mesh.vertices).all()), ""))
        ext = mesh.extents if nf else np.zeros(3)
        checks.append((f"{label} has 3D volume", bool(np.all(ext > 1e-4)), "x".join(f"{e:.3f}" for e in ext)))
        if asset.textured:
            checks.append((f"{label} texture preserved", _has_texture(mesh), ""))
        elif label == "GLB":  # OBJ vertex colours are a de-facto extension; only GLB is guaranteed
            checks.append((f"{label} vertex colours preserved", _has_vertex_colors(mesh), ""))
        if label == "GLB":
            welded = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
            welded.merge_vertices()  # UV seams duplicate vertices; judge closedness on the geometry alone
            stats = {
                "vertices": len(mesh.vertices),
                "faces": nf,
                "watertight": bool(welded.is_watertight),
                "extents_xyz": [round(float(e), 4) for e in ext],
            }
    return {"passed": all(ok for _, ok, _ in checks), "checks": checks, "stats": stats, "depth_scale": bundle.depth_scale, "textured": bundle.textured}


def format_report(report: dict) -> str:
    if not report:
        return ""
    head = "✅ **Export validated**" if report["passed"] else "⚠️ **Export validation found issues**"
    s = report.get("stats", {})
    lines = [head, ""]
    if s:
        ext = s["extents_xyz"]
        lines.append(
            f"{s['vertices']:,} vertices · {s['faces']:,} faces · watertight: {'yes' if s['watertight'] else 'no'} · "
            f"size W×H×D = {ext[0]:.2f} × {ext[1]:.2f} × {ext[2]:.2f} · depth ×{report['depth_scale']:.2f} · "
            f"{'UV texture' if report['textured'] else 'vertex colours'}"
        )
        lines.append("")
    lines += [f"- {'✓' if ok else '✗'} {name}{f' — {detail}' if detail else ''}" for name, ok, detail in report["checks"]]
    return "\n".join(lines)
