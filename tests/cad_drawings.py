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


def pillow_block(path: Path) -> float:
    """Third-angle pillow block drawn like an AutoCAD sheet: OBJECT / HIDDEN / CENTER / DIM / TITLE / BORDER layers
    with BYLAYER linetypes, dimensions, centre lines and a title block. Base 120 x 50 x 15 with two 12 mm bolt
    holes through it; a 60 wide housing with a round top (R30 about x=60, z=45), a 30 mm bore front to back and a
    6 mm grease hole from the top down into the bore."""
    doc, msp = _new()
    doc.layers.add("OBJECT", color=7)
    doc.layers.add("HIDDEN", color=1, linetype="HIDDEN")
    doc.layers.add("CENTER", color=3, linetype="CENTER")
    doc.layers.add("DIM", color=4)
    doc.layers.add("TITLE", color=7)
    obj, hid, cen, dim = ({"layer": n} for n in ("OBJECT", "HIDDEN", "CENTER", "DIM"))
    top_z = 45 + math.sqrt(30**2 - 3**2)  # the arch's height at the grease hole's edge

    # Front (x, z): base plus an arched housing, the bore seen end-on.
    msp.add_lwpolyline([(0, 0, 0), (120, 0, 0), (120, 15, 0), (90, 15, 0), (90, 45, 1), (30, 45, 0), (30, 15, 0), (0, 15, 0)],
                       format="xyb", close=True, dxfattribs=obj)
    msp.add_circle((60, 45), 15, dxfattribs=obj)
    for x in (9, 21, 99, 111):  # bolt holes
        msp.add_line((x, 0), (x, 15), dxfattribs=hid)
    for x in (57, 63):  # grease hole
        msp.add_line((x, 60), (x, top_z), dxfattribs=hid)
    msp.add_line((40, 45), (80, 45), dxfattribs=cen)
    msp.add_line((60, 25), (60, 82), dxfattribs=cen)

    # Top (x, y) above the front: sheet y = 110 + y.
    msp.add_lwpolyline([(0, 110), (120, 110), (120, 160), (0, 160)], close=True, dxfattribs=obj)
    msp.add_line((30, 110), (30, 160), dxfattribs=obj)
    msp.add_line((90, 110), (90, 160), dxfattribs=obj)
    for x, r in ((15, 6), (105, 6), (60, 3)):
        msp.add_circle((x, 135), r, dxfattribs=obj)
        msp.add_line((x - r - 4, 135), (x + r + 4, 135), dxfattribs=cen)
    for x in (45, 75):  # bore
        msp.add_line((x, 110), (x, 160), dxfattribs=hid)

    # Right side (y, z) right of the front: sheet x = 150 + y.
    msp.add_lwpolyline([(150, 0), (200, 0), (200, 75), (150, 75)], close=True, dxfattribs=obj)
    msp.add_line((150, 15), (200, 15), dxfattribs=obj)
    for z in (30, 60):  # bore
        msp.add_line((150, z), (200, z), dxfattribs=hid)
    for y in (19, 31):  # bolt holes (both behind each other)
        msp.add_line((150 + y, 0), (150 + y, 15), dxfattribs=hid)
    for y in (22, 28):  # grease hole
        msp.add_line((150 + y, 60), (150 + y, 75), dxfattribs=hid)
    msp.add_line((145, 45), (205, 45), dxfattribs=cen)

    # Sheet furniture the reader must ignore.
    msp.add_linear_dim(base=(0, -15), p1=(0, 0), p2=(120, 0), dxfattribs=dim).render()
    msp.add_linear_dim(base=(-15, 0), p1=(0, 0), p2=(0, 15), angle=90, dxfattribs=dim).render()
    msp.add_linear_dim(base=(215, 0), p1=(200, 0), p2=(200, 75), angle=90, dxfattribs=dim).render()
    msp.add_diameter_dim(center=(60, 45), radius=15, angle=45, dxfattribs=dim).render()
    msp.add_lwpolyline([(-40, -50), (280, -50), (280, 200), (-40, 200)], close=True, dxfattribs={"layer": "BORDER"})
    msp.add_lwpolyline([(160, -50), (280, -50), (280, -20), (160, -20)], close=True, dxfattribs={"layer": "TITLE"})
    msp.add_text("PILLOW BLOCK", dxfattribs={"layer": "TITLE", "height": 6}).set_placement((165, -32))
    msp.add_text("MS  1:1  mm  3rd ANGLE", dxfattribs={"layer": "TITLE", "height": 3.5}).set_placement((165, -44))
    doc.saveas(path)

    housing = 60 * 30 * 50 + math.pi * 30**2 / 2 * 50
    return 120 * 50 * 15 + housing - math.pi * 15**2 * 50 - 2 * math.pi * 6**2 * 15 - math.pi * 3**2 * (top_z - 60)
