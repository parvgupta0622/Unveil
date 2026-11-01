"""Clean three-view DXF drawings with known solids, for the CAD tests and the bundled example."""
from __future__ import annotations

import math
from pathlib import Path


def _new():
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)  # setup=True adds CENTER / DASHED ...; HIDDEN as in AutoCAD's acad.lin:
    doc.linetypes.add("HIDDEN", pattern=[0.375, 0.25, -0.125], description="Hidden __ __ __")
    for layer in ("FRONT", "TOP", "SIDE", "BORDER", "TEXT"):
        doc.layers.add(layer)
    doc.header["$INSUNITS"] = 4  # millimetres
    return doc, doc.modelspace()


def _poly(msp, pts, layer="0", closed=True):
    msp.add_lwpolyline(pts, close=closed, dxfattribs={"layer": layer})


def _hidden(msp, a, b, layer="0"):
    msp.add_line(a, b, dxfattribs={"layer": layer, "linetype": "HIDDEN"})


def l_bracket(path: Path, layers: bool = False, extras: bool = False) -> float:
    """Third-angle L-bracket: base 100 x 60 x 15, upright 20 x 60 x 50 at the left, a 20 mm through hole in the
    base at x=60, y=30. Returns the solid's volume (mm^3)."""
    doc, msp = _new()
    lf, lt, ls = ("FRONT", "TOP", "SIDE") if layers else ("0", "0", "0")
    # Front (x, z) at the origin.
    _poly(msp, [(0, 0), (100, 0), (100, 15), (20, 15), (20, 50), (0, 50)], lf)
    _hidden(msp, (50, 0), (50, 15), lf)
    _hidden(msp, (70, 0), (70, 15), lf)
    # Top (x, y) above the front: sheet y = 80 + y.
    _poly(msp, [(0, 80), (100, 80), (100, 140), (0, 140)], lt)
    msp.add_line((20, 80), (20, 140), dxfattribs={"layer": lt})
    msp.add_circle((60, 110), 10, dxfattribs={"layer": lt})
    msp.add_line((45, 110), (75, 110), dxfattribs={"layer": lt, "linetype": "CENTER"})
    # Right side (y, z) right of the front: sheet x = 130 + y (rightwards = towards the back).
    _poly(msp, [(130, 0), (190, 0), (190, 50), (130, 50)], ls)
    msp.add_line((130, 15), (190, 15), dxfattribs={"layer": ls})
    _hidden(msp, (150, 0), (150, 15), ls)
    _hidden(msp, (170, 0), (170, 15), ls)
    if extras:  # things the reader must ignore
        _poly(msp, [(-40, -40), (240, -40), (240, 180), (-40, 180)], "BORDER")
        msp.add_text("L-BRACKET  SCALE 1:1", dxfattribs={"layer": "TEXT", "height": 5}).set_placement((0, -30))
    doc.saveas(path)
    return 100 * 60 * 15 + 20 * 60 * 35 - math.pi * 10**2 * 15


def back_wall_first_angle(path: Path) -> float:
    """First-angle part: base 100 x 60 x 15 with a wall 15 thick and 50 high along the BACK edge.
    Plan below the front view, left view to the right of it (rightwards = towards the front)."""
    doc, msp = _new()
    # Front (x, z): the wall is behind the base, so the front view is a 100 x 50 rectangle with the base edge.
    _poly(msp, [(0, 0), (100, 0), (100, 50), (0, 50)])
    msp.add_line((0, 15), (100, 15))
    # Plan below: up = towards the back, so the wall (y 45..60) is the top strip. sheet y = -90 + y.
    _poly(msp, [(0, -90), (100, -90), (100, -30), (0, -30)])
    msp.add_line((0, -45), (100, -45))
    # Left view right of the front: sheet x = 130 + (60 - y), wall drawn next to the front view.
    _poly(msp, [(130, 0), (190, 0), (190, 15), (145, 15), (145, 50), (130, 50)])
    doc.saveas(path)
    return 100 * 60 * 15 + 100 * 15 * 35


def cylinder(path: Path) -> float:
    """Upright cylinder, diameter 40, height 60."""
    doc, msp = _new()
    _poly(msp, [(0, 0), (40, 0), (40, 60), (0, 60)])
    msp.add_circle((20, 100), 20)
    _poly(msp, [(70, 0), (110, 0), (110, 60), (70, 60)])
    doc.saveas(path)
    return math.pi * 20**2 * 60


def plate_with_boss(path: Path) -> float:
    """Plate 80 x 50 x 10 with a solid round boss (diameter 20, 15 tall) on top: the circle in the top view has
    no hidden lines, so it must not be drilled."""
    doc, msp = _new()
    _poly(msp, [(0, 0), (80, 0), (80, 10), (50, 10), (50, 25), (30, 25), (30, 10), (0, 10)])
    _poly(msp, [(0, 60), (80, 60), (80, 110), (0, 110)])
    msp.add_circle((40, 85), 10)
    _poly(msp, [(110, 0), (160, 0), (160, 10), (145, 10), (145, 25), (125, 25), (125, 10), (110, 10)])
    doc.saveas(path)
    return 80 * 50 * 10 + math.pi * 10**2 * 15
