"""Checks the TripoSR adapter against TripoSR's own code (no weights needed).

Requires the TripoSR source (scripts/setup_colab.sh clones it, or set UNVEIL_TRIPOSR_PATH).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from unveil.reconstruction import triposr as T  # noqa: E402


def _tsr_path():
    path = Path(os.environ.get("UNVEIL_TRIPOSR_PATH") or T.DEFAULT_REPO_PATH)
    return path if (path / "tsr" / "system.py").exists() else None


def test_native_to_view_is_a_rotation():
    R = T.NATIVE_TO_VIEW
    np.testing.assert_allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)
    # TripoSR: camera on +x, +y right, +z up  ->  view: +z to camera, +x right, +y up
    np.testing.assert_array_equal(R @ [1, 0, 0], [0, 0, 1])
    np.testing.assert_array_equal(R @ [0, 1, 0], [1, 0, 0])
    np.testing.assert_array_equal(R @ [0, 0, 1], [0, 1, 0])


@pytest.mark.skipif(_tsr_path() is None, reason="TripoSR source not available")
def test_marching_cubes_shim_matches_triposr_grid(monkeypatch):
    if "torchmcubes" in sys.modules and not getattr(sys.modules["torchmcubes"], "__unveil_shim__", False):
        pytest.skip("real torchmcubes installed; shim not in use")
    T.ensure_triposr_importable(str(_tsr_path()))
    from tsr.models.isosurface import MarchingCubeHelper

    helper = MarchingCubeHelper(64)
    c = helper.grid_vertices - 0.5  # (N, 3), same ordering TripoSR samples density in
    density = 1 - ((c[:, 0] / 0.40) ** 2 + (c[:, 1] / 0.15) ** 2 + (c[:, 2] / 0.25) ** 2)
    v, f = helper(-density)  # TSR.extract_mesh passes -(density - threshold)
    ext = (v.max(0).values - v.min(0).values).numpy()
    # The surface must come back in the same axes the density was sampled in.
    np.testing.assert_allclose(ext, [0.8, 0.3, 0.5], atol=0.04)
    assert f.dtype == torch.int64 and len(f) > 100


@pytest.mark.skipif(_tsr_path() is None, reason="TripoSR source not available")
def test_shim_empty_volume():
    T.ensure_triposr_importable(str(_tsr_path()))
    import torchmcubes

    v, f = torchmcubes.marching_cubes(torch.zeros(8, 8, 8), 0.5)
    assert v.shape == (0, 3) and f.shape == (0, 3)
