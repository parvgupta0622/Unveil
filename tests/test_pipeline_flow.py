"""End-to-end orchestration test on CPU.

The three neural models (SD+ControlNet, CLIP, TripoSR) are replaced by small deterministic stand-ins so
that every other part of the real pipeline (preprocessing, background removal, scoring, ranking,
candidate cleanup and verification, hollowing, texture bake, export, validation, depth re-export) runs for real.
"""
import json
import zipfile

import cv2
import numpy as np
import pytest
import trimesh
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

pytest.importorskip("torch")

from tests.conftest import draw_outline  # noqa: E402
from unveil.background import composite_on, resize_foreground  # noqa: E402
from unveil.config import UnveilConfig  # noqa: E402
from unveil.mesh.hollow import HOLLOW_OBJECTS, SOLID_OBJECTS  # noqa: E402
from unveil.pipeline import GuideViews, ImageRun, PipelineError, RunOptions, RunResult, StageEvent, UnveilPipeline  # noqa: E402
from unveil.reconstruction.base import CameraSpec, ReconstructionResult, Reconstructor  # noqa: E402


class StandInGenerator:
    loaded, device = True, "cpu"

    def to(self, device):
        pass

    def load(self, mode):
        pass

    def generate(self, control, prompt, negative, level, seed, mode, steps, guidance, step_callback=None):
        lines = np.asarray(control.convert("L")) > 0
        mask = ndi.binary_fill_holes(lines)
        grow = int(1 + 40 * (1.3 - level.conditioning_scale))  # looser levels drift from the sketch
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0
        img = np.full(mask.shape + (3,), 250, np.uint8)
        img[mask] = (180, 60, 50)
        return Image.fromarray(img)


class StandInClip:
    loaded, device = True, "cpu"

    def to(self, device):
        pass

    def similarity(self, images, prompt, reference):
        return [(0.5, 0.25)] * len(images)

    def zero_shot(self, image, labels):  # always "chair": solid
        probs = np.full(len(labels), 0.01 / len(labels))
        probs[labels.index("chair")] += 0.99
        return probs


class ExtrusionReconstructor(Reconstructor):
    """Extrudes the foreground mask into a slab, plus a floating speck to exercise cleanup."""

    name = "extrusion"
    camera = CameraSpec(distance=1.9, fov_deg=40.0)
    device = "cpu"

    @property
    def loaded(self):
        return True

    def load(self):
        pass

    def to(self, device):
        pass

    def prepare_input(self, rgba, ratio):
        return composite_on(resize_foreground(rgba, ratio))

    def reconstruct(self, image, mc_resolution, mc_threshold):
        arr = np.asarray(image.resize((64, 64))).astype(int)
        mask = np.abs(arr - 128).sum(-1) > 40
        vol = np.zeros((64, 64, 24))
        vol[mask, 4:20] = 1.0
        vol[2:4, 2:4, 10:12] = 1.0  # floater
        verts, faces, _, _ = marching_cubes(np.pad(vol, 1), 0.5)
        row, col, dep = verts[:, 0] - 1, verts[:, 1] - 1, verts[:, 2] - 1
        view = np.stack([(col - 32) / 64, (32 - row) / 64, (dep - 12) / 64], axis=-1)
        mesh = trimesh.Trimesh(view, faces, vertex_colors=np.tile([200, 80, 60, 255], (len(view), 1)), process=False)
        return ReconstructionResult(mesh=mesh, handle="scene-code")

    @property
    def supports_color_query(self):
        return True

    def query_color(self, handle, points):
        assert handle == "scene-code"
        return np.tile([0.8, 0.3, 0.2], (len(points), 1))


@pytest.fixture
def pipe(tmp_path):
    p = UnveilPipeline(UnveilConfig(device="cpu", output_dir=str(tmp_path)))
    p.generator, p.clip, p.reconstructor = StandInGenerator(), StandInClip(), ExtrusionReconstructor()
    return p


def _run(pipe, **kw):
    events = list(pipe.run(draw_outline("L"), RunOptions(texture_resolution=256, **kw)))
    assert isinstance(events[-1], RunResult)
    return events[:-1], events[-1]


def test_full_run(pipe):
    events, result = _run(pipe)
    assert all(isinstance(e, StageEvent) for e in events)
    done = [e.stage for e in events if e.status == "done"]
    assert done == ["preprocess", "generate", "select_image", "reconstruct", "interior", "refine", "export"]

    s = result.summary
    assert len(s["variations"]) == 4 and len(s["candidates"]) == 4
    # Ranking follows the combined score; the variation that drifts furthest from the sketch ranks last.
    totals = [v["total"] for v in s["variations"]]
    assert s["variation_ranking"][0] == int(np.argmax(totals))
    assert s["variations"][s["variation_ranking"][-1]]["level"] == "free"
    assert s["selection"] == {"variations": s["variation_ranking"][:2], "selected_by": "auto"}

    # One final model per selected variation, each its variation's best verified candidate.
    assert [o.variation for o in result.outputs] == s["variation_ranking"][:2]
    for o in result.outputs:
        best = s["candidates"][o.candidate]
        assert best["variation"] == o.variation
        assert best["score"]["valid"] and best["cleanup"]["retention"] < 1.0  # floater was removed
        assert best["score"]["iou"] > 0.8
        assert o.bundle.report["passed"], o.bundle.report["checks"]
        assert o.bundle.textured
        assert not o.interior.hollow and o.interior.source == "clip"  # stand-in CLIP says "chair"
    assert result.outputs[0].bundle.glb != result.outputs[1].bundle.glb

    # Every model in every format ends up in the combined download.
    names = zipfile.ZipFile(result.all_zip).namelist()
    for o in result.outputs:
        assert o.bundle.glb.name in names and f"{o.bundle.obj.parent.name}/{o.bundle.obj.name}" in names
    assert (result.run_dir / "run.json").exists()
    assert json.loads((result.run_dir / "run.json").read_text())["run_id"] == result.run_id


def test_user_picks_variation(pipe):
    events = list(pipe.run_images(draw_outline("L"), RunOptions(texture_resolution=256, bake_texture=False)))
    img = events[-1]
    assert isinstance(img, ImageRun) and not any(e.stage == "reconstruct" for e in events[:-1])
    worst = img.ranking[-1]
    out = list(pipe.run_3d(img.run_id, [worst]))
    result = out[-1]
    assert isinstance(result, RunResult)
    assert [o.variation for o in result.outputs] == [worst] and result.summary["selection"]["selected_by"] == "user"
    with pytest.raises(PipelineError):
        list(pipe.run_3d(img.run_id, [0, 1, 2]))  # at most two final models
    with pytest.raises(PipelineError):
        list(pipe.run_3d("no-such-run", [0]))


def test_hollow_vessel_gets_a_cavity(pipe):
    _, solid = _run(pipe, bake_texture=False, interior="solid", top_k_images=1)
    _, mug = _run(pipe, bake_texture=False, prompt="a red ceramic mug", top_k_images=1)
    o = mug.outputs[0]
    assert o.interior.hollow and o.interior.source == "prompt" and o.interior.label == "mug"
    assert o.hollow is not None and o.hollow.opened and o.hollow.cavity_fraction > 0.2
    assert o.bundle.report["passed"], o.bundle.report["checks"]
    solid_vol = trimesh.load(solid.outputs[0].bundle.glb, force="mesh").volume
    hollow_vol = trimesh.load(o.bundle.glb, force="mesh").volume
    assert hollow_vol < 0.8 * solid_vol
    # The cross-section through the middle shows the empty cavity inside the walls.
    section = np.asarray(o.section.convert("L")) < 200
    assert section.sum() < 0.8 * (np.asarray(solid.outputs[0].section.convert("L")) < 200).sum()


def test_depth_reexport_changes_geometry(pipe):
    _, result = _run(pipe, bake_texture=False)
    base = [o.bundle.report["stats"]["extents_xyz"] for o in result.outputs]
    thin, zip_path = pipe.reexport(result.build_dir, 0.5)
    thick, _ = pipe.reexport(result.build_dir, 1.6)
    assert len(thin) == len(result.outputs) and zip_path.exists()
    for b, t, k in zip(base, thin, thick):
        t, k = t.report["stats"]["extents_xyz"], k.report["stats"]["extents_xyz"]
        assert t[0] == pytest.approx(b[0]) and t[1] == pytest.approx(b[1])
        assert t[2] == pytest.approx(b[2] * 0.5, rel=0.02)
        assert k[2] == pytest.approx(b[2] * 1.6, rel=0.02)
    reloaded = trimesh.load(thin[0].glb, force="mesh")
    assert reloaded.extents[2] == pytest.approx(thin[0].report["stats"]["extents_xyz"][2], rel=0.01)


def test_bake_after_run(pipe):
    _, result = _run(pipe, bake_texture=False, prompt="a vase")
    assert not result.bundle.textured
    bundles, _, warnings = pipe.bake_existing(result.build_dir, 1.0, 256)
    assert all(b.textured and b.report["passed"] for b in bundles) and not warnings


def test_front_back_symmetry_with_hollow_and_bake(pipe):
    _, result = _run(pipe, front_back_symmetric=True, prompt="a red ceramic mug", top_k_images=1)
    o = result.outputs[0]
    assert o.symmetry is not None and o.hollow is not None
    assert o.bundle.textured and o.bundle.report["passed"], o.bundle.report["checks"]
    assert result.summary["outputs"][0]["symmetry"]["plane_z"] == pytest.approx(0.0, abs=0.03)  # slab centred on z = 0


def _side_sketch(depth_ratio):
    """A side view as a user would draw it: a rectangle outline as tall as the object and `depth_ratio` x as deep."""
    img = np.full((512, 512), 255, np.uint8)
    h = 330
    cv2.rectangle(img, (100, 90), (100 + int(h * depth_ratio), 90 + h), 0, 4)
    return Image.fromarray(img)


def test_side_view_guide_trims_depth_and_heatmap_is_exported(pipe):
    _, plain = _run(pipe, bake_texture=False, top_k_images=1)
    assert plain.outputs[0].guide is None and plain.outputs[0].confidence is not None
    assert plain.outputs[0].bundle.heatmap.exists()  # every sketch run gets a confidence heatmap
    events = list(pipe.run(draw_outline("L"), RunOptions(texture_resolution=256, top_k_images=1), GuideViews(side=_side_sketch(0.15))))
    result = events[-1]
    o = result.outputs[0]
    assert o.guide is not None and o.guide.views == ["side"] and o.guide.trimmed_fraction > 0.2
    assert o.guide.depth_after < 0.7 * o.guide.depth_before
    assert o.bundle.report["passed"], o.bundle.report["checks"]
    assert o.bundle.textured  # colours are still baked from TripoSR's field after trimming
    assert o.confidence.guided > 0.2 and o.bundle.heatmap.exists()
    assert any("side view guide" in caption for _, caption in result.candidate_gallery)
    assert result.summary["outputs"][0]["guide"]["views"] == ["side"] and result.summary["guide_views"]["given"] == ["side"]
    thin, _ = pipe.reexport(result.build_dir, 0.5)  # the heatmap follows a depth change
    assert thin[0].heatmap is not None and "depth0.50" in thin[0].heatmap.name and thin[0].heatmap.exists()
    with pytest.raises(PipelineError) as err:  # a blank guide canvas that slipped through fails before TripoSR
        list(pipe.run_3d(result.run_id, [0], guides=GuideViews(top=Image.new("RGB", (256, 256), "white"))))
    assert err.value.stage == "interior" and "top-view sketch" in str(err.value)


def test_cad_drawing_run(pipe, tmp_path):
    pytest.importorskip("ezdxf")
    from tests.cad_drawings import l_bracket
    from unveil.pipeline import CadRun

    path = tmp_path / "bracket.dxf"
    l_bracket(path)
    events = list(pipe.run_cad(path))
    run = events[-1]
    assert isinstance(run, CadRun)
    assert [e.stage for e in events[:-1] if e.status == "done"] == ["cad_read", "cad_views", "cad_solid", "export"]
    assert any("cad_images" in e.payload for e in events[:-1])
    assert run.bundle.report["passed"], run.bundle.report["checks"]
    assert run.bundle.report["stats"]["extents_xyz"] == pytest.approx([100, 50, 60], rel=0.02)  # x, height, depth
    thin, _ = pipe.reexport(run.build_dir, 0.5)  # the depth control works on CAD models too
    assert thin[0].report["stats"]["extents_xyz"][2] == pytest.approx(30, rel=0.02)

    import ezdxf

    doc = ezdxf.new()
    doc.modelspace().add_circle((0, 0), 5)
    doc.saveas(tmp_path / "one.dxf")
    with pytest.raises(PipelineError) as err:
        list(pipe.run_cad(tmp_path / "one.dxf"))
    assert err.value.stage == "cad_views" and "three views" in str(err.value) and "CadError" not in str(err.value)


def test_category_lists_do_not_overlap():
    assert not set(HOLLOW_OBJECTS) & set(SOLID_OBJECTS)


def test_ui_builds(tmp_path):
    gr = pytest.importorskip("gradio")
    from unveil.ui import build_demo

    assert isinstance(build_demo(UnveilConfig(device="cpu", output_dir=str(tmp_path))), gr.Blocks)
