"""Guide views (trim to side/top sketches, open see-through holes) and the confidence heatmap."""
import cv2
import numpy as np
import pytest
import trimesh

from unveil.mesh import MeshAsset, apply_guides, export_heatmap, save_confidence, see_through_holes, surface_confidence
from unveil.reconstruction.base import CameraSpec

CAM = CameraSpec(distance=1.9, fov_deg=40.0)


def _box(extents, center=(0, 0, 0)):
    box = trimesh.creation.box(extents)
    v, f = trimesh.remesh.subdivide_to_size(box.vertices, box.faces, 0.05)
    mesh = trimesh.Trimesh(v + np.asarray(center), f, process=False)
    mesh.visual.vertex_colors = np.tile([200, 80, 60, 255], (len(v), 1))
    return mesh


def _l_profile(front_left=True):
    """Side outline of an L: 0.6 tall and 0.3 deep, with a 0.2 tall foot 0.5 deep (2.5 mm per px at 0.6 = 240 px)."""
    side = np.zeros((300, 300), bool)
    side[30:270, 50:170] = True
    side[190:270, 50:250] = True
    return side if front_left else side[:, ::-1].copy()


def test_side_view_trims_the_depth_and_keeps_the_front():
    guessed = _box((0.6, 0.6, 0.8))  # TripoSR made it too deep; the front is at z = +0.4
    res, reason = apply_guides(guessed, CAM, side=_l_profile(), resolution=128)
    assert res is not None and res.changed, reason
    v = res.mesh.vertices
    assert v[:, 2].max() == pytest.approx(0.4, abs=0.02)  # front kept where it was
    assert v[:, 2].min() == pytest.approx(-0.1, abs=0.04)  # 0.5 deep foot (+ the outline leeway)
    assert res.mesh.is_watertight
    assert res.mesh.volume == pytest.approx(0.6 * (0.6 * 0.3 + 0.2 * 0.2), rel=0.1)
    assert res.stats.views == ["side"] and res.stats.depth_after < res.stats.depth_before
    assert len(res.overlays) == 1
    # The same outline drawn facing the other way gives the same solid.
    mirrored, _ = apply_guides(guessed, CAM, side=_l_profile(front_left=False), side_front_left=False, resolution=128)
    assert mirrored.mesh.volume == pytest.approx(res.mesh.volume, rel=0.01)


def test_top_view_trims_and_a_fitting_view_changes_nothing():
    guessed = _box((0.6, 0.6, 0.6))
    top = np.zeros((300, 300), np.uint8)
    cv2.fillConvexPoly(top, np.array([[30, 270], [270, 270], [210, 30], [90, 30]], np.int32), 1)  # narrower at the back
    res, _ = apply_guides(guessed, CAM, top=top.astype(bool), resolution=128)
    assert res.changed and res.mesh.volume == pytest.approx(0.6 * 0.6 * 0.6 * 0.75, rel=0.1)
    square = np.zeros((300, 300), bool)
    square[30:270, 30:270] = True
    same, _ = apply_guides(guessed, CAM, side=square, top=square, resolution=128)
    assert not same.changed and same.mesh is guessed and same.stats.trimmed_fraction < 0.01


def test_a_side_view_that_does_not_match_is_refused():
    sliver = np.zeros((300, 300), bool)
    sliver[30:270, 50:54] = True
    sliver[150:154, 50:290] = True  # a thin cross: almost all of the model would go
    res, reason = apply_guides(_box((0.6, 0.6, 0.6)), CAM, side=sliver, resolution=96)
    assert res is None and "hardly overlap" in reason


def _holed_square(with_loop=True):
    """512 px image of a square plate with a square see-through hole, and the matching sketch ink."""
    fg = np.zeros((512, 512), np.uint8)
    fg[106:406, 106:406] = 255
    fg[206:306, 206:306] = 0
    ink = np.zeros((512, 512), np.uint8)
    cv2.rectangle(ink, (106, 106), (405, 405), 1, 3)
    if with_loop:
        cv2.rectangle(ink, (206, 206), (305, 305), 1, 3)
    return fg, ink.astype(bool)


def test_see_through_hole_needs_the_image_and_the_sketch_to_agree():
    fg, ink = _holed_square()
    holes, _ = see_through_holes(fg, ink)
    assert holes[256, 256] and not holes[150, 150] and 0.03 < holes.mean() < 0.045
    fg_only, ink_no_loop = _holed_square(with_loop=False)
    assert not see_through_holes(fg_only, ink_no_loop)[0].any()  # e.g. a bright highlight: not drilled


def test_hole_is_drilled_through_a_filled_plate():
    plate = _box((0.6, 0.6, 0.2))  # TripoSR closed the hole
    fg, ink = _holed_square()
    res, reason = apply_guides(plate, CAM, holes=see_through_holes(fg, ink), resolution=128)
    assert res is not None and res.changed, reason
    v = res.mesh.vertices
    assert not ((np.abs(v[:, 0]) < 0.05) & (np.abs(v[:, 1]) < 0.05)).any()  # nothing left in the hole
    assert res.stats.holes == 1 and 0.05 < res.stats.hole_fraction < 0.12
    assert res.mesh.is_watertight


def test_confidence_follows_agreement_visibility_and_guides():
    ball = trimesh.creation.icosphere(4, 0.4)
    bigger = ball.copy()
    bigger.apply_scale(1.15)  # another framing of the same object: alignment removes the scale
    deep = ball.copy()
    deep.vertices = np.where(ball.vertices[:, 2:] < 0, ball.vertices * [1, 1, 1.6], ball.vertices)
    front, back = ball.vertices[:, 2] > 0.2, ball.vertices[:, 2] < -0.2
    agree = surface_confidence(ball, ball, [bigger, bigger], CAM)
    assert agree.values[front].mean() < 0.1 and 0.2 < agree.values[back].mean() < 0.5  # back unseen, but agreed
    differ = surface_confidence(ball, ball, [deep], CAM)
    assert differ.values[back].mean() > 0.8 and differ.stats.guess > 0.2
    guided = surface_confidence(ball, ball, [deep], CAM, pinned=lambda p: p[:, 2] < 0)
    assert guided.values[back].mean() < 0.5 and guided.stats.guided > 0.3
    single = surface_confidence(ball, ball, [], CAM)
    assert single.stats.attempts == 0 and single.values[back].mean() == pytest.approx(0.5, abs=0.05)


def test_heatmap_export_survives_reindexed_vertices(tmp_path):
    ball = trimesh.creation.icosphere(3, 0.4)
    asset = MeshAsset.from_trimesh(ball)
    values = (asset.vertices[:, 2] < 0).astype(np.float32)  # back = guess
    save_confidence(tmp_path / "confidence_model1.npz", asset.vertices, values)
    # A bake duplicates seam vertices and re-orders them; the lookup is by position.
    perm = np.random.default_rng(0).permutation(len(asset.vertices))
    inv = np.argsort(perm)
    shuffled = MeshAsset(asset.vertices[perm], inv[asset.faces])
    path = export_heatmap(shuffled, tmp_path / "confidence_model1.npz", tmp_path / "heat.obj", depth_scale=0.5)
    lines = path.read_text().splitlines()
    verts = np.array([[float(x) for x in ln.split()[1:]] for ln in lines if ln.startswith("v ")])
    assert len(verts) == len(asset.vertices) and sum(ln.startswith("f ") for ln in lines) == len(asset.faces)
    back = verts[:, 2] < -0.01
    assert (verts[back, 3] > 0.8).all() and (verts[~back & (verts[:, 2] > 0.01), 4] > 0.6).all()  # red back, green front
    assert np.ptp(verts[:, 2]) == pytest.approx(0.4, rel=0.05)  # depth scale applied like the model's export
    assert export_heatmap(shuffled, tmp_path / "missing.npz", tmp_path / "x.obj") is None
