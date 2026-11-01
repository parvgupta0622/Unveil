"""CAD drawing -> 3D: three-view DXF drawings with known solids (and DWG when LibreDWG is installed)."""
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("ezdxf")

from .cad_drawings import back_wall_first_angle, cylinder, l_bracket, plate_with_boss  # noqa: E402

from unveil.cad import CadError, cad_to_model  # noqa: E402
from unveil.mesh.processing import signed_volume  # noqa: E402


def _volume(model):
    return signed_volume(model.mesh.vertices, model.mesh.faces)


def test_third_angle_bracket_with_through_hole(tmp_path):
    path = tmp_path / "bracket.dxf"
    expected = l_bracket(path)
    model = cad_to_model(path)
    assert model.projection == "third-angle" and model.view_source == "layout"
    np.testing.assert_allclose(model.dimensions, (100, 60, 50), rtol=0.01)
    assert [(h.view, h.evidence) for h in model.holes] == [("top", "front")]
    assert model.holes[0].depth_range == pytest.approx((0, 15), abs=0.5)
    assert model.mesh.is_watertight
    assert _volume(model) == pytest.approx(expected, rel=0.03)
    assert model.units == "Millimeters"


def test_views_from_layer_names_ignore_border_and_text(tmp_path):
    path = tmp_path / "bracket_layers.dxf"
    expected = l_bracket(path, layers=True, extras=True)
    model = cad_to_model(path)
    assert model.view_source == "layers"
    assert _volume(model) == pytest.approx(expected, rel=0.03)


def test_first_angle_side_view_is_mirrored(tmp_path):
    """First-angle left view: sheet-rightwards is towards the FRONT. The wall must end up at the back."""
    path = tmp_path / "back_wall.dxf"
    expected = back_wall_first_angle(path)
    model = cad_to_model(path)
    assert model.projection == "first-angle"
    assert _volume(model) == pytest.approx(expected, rel=0.03)
    mesh = model.mesh  # view frame: z = 0 is the front face, the back is at z = -60
    tall = mesh.vertices[mesh.vertices[:, 1] > 40]
    assert tall[:, 2].max() < -40  # the 50 mm tall wall sits in the back 15 mm


def test_cylinder_and_boss_without_hidden_lines_stay_solid(tmp_path):
    cyl_path = tmp_path / "cyl.dxf"
    expected = cylinder(cyl_path)
    assert _volume(cad_to_model(cyl_path)) == pytest.approx(expected, rel=0.03)
    boss_path = tmp_path / "boss.dxf"
    expected = plate_with_boss(boss_path)
    boss = cad_to_model(boss_path)
    assert not boss.holes and _volume(boss) == pytest.approx(expected, rel=0.04)
    assert any("hidden" in w for w in boss.warnings)


def test_unusable_drawings_give_clear_errors(tmp_path):
    import ezdxf

    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline([(0, 0), (10, 0), (10, 10)], close=True)
    one_view = tmp_path / "one.dxf"
    doc.saveas(one_view)
    with pytest.raises(CadError, match="three views"):
        cad_to_model(one_view)
    with pytest.raises(CadError, match=".dxf or .dwg"):
        cad_to_model(tmp_path / "drawing.pdf")


@pytest.mark.skipif(shutil.which("dwg2dxf") is None or shutil.which("dxf2dwg") is None, reason="LibreDWG not installed")
def test_dwg_is_converted(tmp_path):
    dxf, dwg = tmp_path / "bracket.dxf", tmp_path / "bracket.dwg"
    expected = l_bracket(dxf)
    subprocess.run(["dxf2dwg", "-y", "-o", str(dwg), str(dxf)], check=True, capture_output=True)
    model = cad_to_model(dwg)
    assert len(model.holes) == 1 and _volume(model) == pytest.approx(expected, rel=0.03)
