import zipfile

import numpy as np
import trimesh

from unveil.config import VerifyConfig
from unveil.mesh import MeshAsset, bake_texture, clean_mesh, decimate, export_asset, smooth_mesh
from unveil.mesh.processing import signed_volume
from unveil.mesh.verify import normalize_mask, project, render_silhouette, score_candidate
from unveil.reconstruction.base import CameraSpec
from unveil.scoring import mask_iou

CAM = CameraSpec(distance=1.9, fov_deg=40.0)


def _colored(mesh):
    rng = np.random.default_rng(0)
    mesh.visual.vertex_colors = rng.integers(0, 255, (len(mesh.vertices), 4), dtype=np.uint8)
    return mesh


def test_clean_keeps_largest_component_and_fixes_winding():
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.4)
    floater = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    floater.apply_translation((0.7, 0.7, 0.0))
    combined = trimesh.util.concatenate([sphere, floater])
    inverted = trimesh.Trimesh(combined.vertices, combined.faces[:, ::-1], process=False)
    cleaned, stats = clean_mesh(_colored(inverted))
    assert len(cleaned.faces) == len(sphere.faces)
    assert stats.components == 2 and stats.retention < 1.0
    assert stats.flipped and signed_volume(cleaned.vertices, cleaned.faces) > 0
    assert cleaned.visual.kind == "vertex" and len(cleaned.visual.vertex_colors) == len(cleaned.vertices)


def test_smooth_and_decimate_keep_colors():
    mesh = _colored(trimesh.creation.icosphere(subdivisions=5, radius=0.4))
    smoothed = smooth_mesh(mesh, 3)
    assert smoothed.vertices.shape == mesh.vertices.shape
    small = decimate(mesh, 2000)
    assert len(small.faces) <= 2100
    assert small.visual.kind == "vertex"


def test_depth_scale_changes_only_z():
    box = MeshAsset.from_trimesh(trimesh.creation.box(extents=(0.4, 0.6, 0.2)))
    for s in (0.5, 1.6):
        ext = box.with_depth_scale(s).extents()
        np.testing.assert_allclose(ext, [0.4, 0.6, 0.2 * s], atol=1e-5)


def test_projection_orientation():
    # View frame: +x right, +y up, +z toward the camera => image right / image up.
    px = project(np.array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.0]]), CAM, 256)
    centre = px[2]
    assert px[0, 0] > centre[0] + 10 and abs(px[0, 1] - centre[1]) < 1
    assert px[1, 1] < centre[1] - 10 and abs(px[1, 0] - centre[0]) < 1


def test_silhouette_score_prefers_matching_shape():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    render = render_silhouette(sphere.vertices, sphere.faces, CAM, 256)
    yy, xx = np.mgrid[:256, :256]
    disk = (xx - 128) ** 2 + (yy - 128) ** 2 < 90**2
    square = (abs(xx - 128) < 90) & (abs(yy - 128) < 90)
    assert mask_iou(normalize_mask(render), normalize_mask(disk)) > 0.9

    _, stats = clean_mesh(sphere)
    good, overlay = score_candidate(sphere, disk, CAM, stats, VerifyConfig())
    bad, _ = score_candidate(sphere, square, CAM, stats, VerifyConfig())
    assert good.valid and good.total > bad.total
    assert overlay.size == (256, 256)


def test_flat_mesh_is_invalid():
    card = trimesh.creation.box(extents=(0.6, 0.6, 0.005))
    yy, xx = np.mgrid[:256, :256]
    square = (abs(xx - 128) < 90) & (abs(yy - 128) < 90)
    _, stats = clean_mesh(card)
    cfg = VerifyConfig(min_faces=1)
    score, _ = score_candidate(card, square, CAM, stats, cfg)
    assert not score.valid and "flat" in score.reason


def test_vertex_color_export_roundtrip(tmp_path):
    asset = MeshAsset.from_trimesh(_colored(trimesh.creation.icosphere(subdivisions=3, radius=0.4)))
    asset.save(tmp_path / "asset.npz")
    loaded = MeshAsset.load(tmp_path / "asset.npz")
    np.testing.assert_array_equal(loaded.faces, asset.faces)
    bundle = export_asset(loaded, tmp_path, depth_scale=1.6)
    assert bundle.report["passed"], bundle.report["checks"]
    assert abs(bundle.report["stats"]["extents_xyz"][2] - 0.8 * 1.6) < 0.02
    with zipfile.ZipFile(bundle.obj_zip) as zf:
        assert any(n.endswith(".obj") for n in zf.namelist())


def test_texture_bake_and_textured_export(tmp_path):
    asset = MeshAsset.from_trimesh(trimesh.creation.icosphere(subdivisions=3, radius=0.4))

    def color_fn(points):  # red on the right half, blue on the left
        rgb = np.zeros((len(points), 3))
        rgb[points[:, 0] > 0, 0] = 1.0
        rgb[points[:, 0] <= 0, 2] = 1.0
        return rgb

    textured = bake_texture(asset, color_fn, resolution=256)
    assert textured.textured and textured.texture.shape == (256, 256, 3)
    assert textured.uv.min() >= 0 and textured.uv.max() <= 1
    tex = textured.texture.reshape(-1, 3)
    assert (tex[:, 0] > 200).mean() > 0.2 and (tex[:, 2] > 200).mean() > 0.2
    assert (tex.sum(1) == 0).mean() < 0.01  # empty texels were dilated

    # Sample the texture through the UVs at a right-side vertex: it must be red.
    i = int(np.argmax(textured.vertices[:, 0]))
    u, v = textured.uv[i]
    texel = textured.texture[min(255, int((1 - v) * 256)), min(255, int(u * 256))]
    assert texel[0] > 200 and texel[2] < 50

    bundle = export_asset(textured, tmp_path)
    assert bundle.report["passed"], bundle.report["checks"]
    with zipfile.ZipFile(bundle.obj_zip) as zf:
        names = zf.namelist()
    assert any(n.endswith(".mtl") for n in names) and any(n.endswith(".png") for n in names)
