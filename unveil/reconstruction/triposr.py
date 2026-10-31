"""TripoSR backend (https://github.com/VAST-AI-Research/TripoSR, MIT licence).

TripoSR is not pip-installable, so its source is cloned into third_party/TripoSR. Its marching cubes
depends on `torchmcubes`, a CUDA extension that must be compiled; when that is not installed we register
a drop-in shim backed by scikit-image so no compilation step is needed.
"""
from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh
from PIL import Image

from ..background import composite_on, resize_foreground
from ..config import REPO_ROOT
from ..utils import LOG
from .base import CameraSpec, ReconstructionError, ReconstructionResult, Reconstructor

TRIPOSR_GIT = "https://github.com/VAST-AI-Research/TripoSR.git"
TRIPOSR_COMMIT = "107cefdc244c39106fa830359024f6a2f1c78871"
DEFAULT_REPO_PATH = REPO_ROOT / "third_party" / "TripoSR"

# TripoSR native frame: input camera on +x, +y right, +z up. View frame: +x right, +y up, +z to camera.
# view = R @ native  (a proper rotation, det = +1)
NATIVE_TO_VIEW = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)


def install_torchmcubes_shim() -> str:
    """Make `import torchmcubes` work. Returns which implementation is active."""
    try:
        import torchmcubes  # noqa: F401

        return "skimage-shim" if getattr(torchmcubes, "__unveil_shim__", False) else "torchmcubes"
    except ImportError:
        pass

    import torch

    def marching_cubes(vol, thresh):
        from skimage.measure import marching_cubes as sk_marching_cubes

        arr = vol.detach().float().cpu().numpy()
        if not (arr.min() < thresh < arr.max()):
            return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0, 3), dtype=torch.int64)
        verts, faces, _, _ = sk_marching_cubes(arr, level=float(thresh))
        # torchmcubes returns (x, y, z) = (dim2, dim1, dim0); skimage returns index order (dim0, dim1, dim2).
        verts = np.ascontiguousarray(verts[:, ::-1], dtype=np.float32)
        return torch.from_numpy(verts), torch.from_numpy(faces.astype(np.int64))

    shim = types.ModuleType("torchmcubes")
    shim.marching_cubes = marching_cubes
    shim.__unveil_shim__ = True
    sys.modules["torchmcubes"] = shim
    LOG.info("torchmcubes not installed - using scikit-image marching cubes shim")
    return "skimage-shim"


# transformers 5 restructured ViTModel (`encoder.layer.N.attention.attention.query` ->
# `layers.N.attention.q_proj`, ...). TripoSR's checkpoint uses the transformers-4 names, so they are
# renamed on load when the model expects the new layout. Most specific rules first.
_VIT_V5_RENAMES = [
    (re.compile(r"\.encoder\.layer\.(\d+)\.attention\.attention\.query\."), r".layers.\1.attention.q_proj."),
    (re.compile(r"\.encoder\.layer\.(\d+)\.attention\.attention\.key\."), r".layers.\1.attention.k_proj."),
    (re.compile(r"\.encoder\.layer\.(\d+)\.attention\.attention\.value\."), r".layers.\1.attention.v_proj."),
    (re.compile(r"\.encoder\.layer\.(\d+)\.attention\.output\.dense\."), r".layers.\1.attention.o_proj."),
    (re.compile(r"\.encoder\.layer\.(\d+)\.intermediate\.dense\."), r".layers.\1.mlp.fc1."),
    (re.compile(r"\.encoder\.layer\.(\d+)\.output\.dense\."), r".layers.\1.mlp.fc2."),
    (re.compile(r"\.encoder\.layer\.(\d+)\."), r".layers.\1."),
]


def adapt_state_dict(state: dict, expected_keys) -> dict:
    """Rename checkpoint keys that only differ from the model's by the transformers-5 ViT layout."""
    expected = set(expected_keys)
    if expected.issubset(state):
        return state
    out = {}
    for key, value in state.items():
        if key not in expected:
            for pattern, repl in _VIT_V5_RENAMES:
                renamed = pattern.sub(repl, key)
                if renamed != key:
                    if renamed in expected:
                        key = renamed
                    break
        out[key] = value
    return out


def load_tsr(model_id: str):
    """Equivalent of TSR.from_pretrained, tolerant of transformers 4/5 key layouts."""
    import torch
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from tsr.system import TSR

    if os.path.isdir(model_id):
        config_path, weight_path = os.path.join(model_id, "config.yaml"), os.path.join(model_id, "model.ckpt")
    else:
        config_path = hf_hub_download(repo_id=model_id, filename="config.yaml")
        weight_path = hf_hub_download(repo_id=model_id, filename="model.ckpt")
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    model = TSR(cfg)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(adapt_state_dict(state, model.state_dict().keys()))
    return model


def ensure_triposr_importable(repo_path: Optional[str] = None) -> Path:
    path = Path(repo_path or os.environ.get("UNVEIL_TRIPOSR_PATH") or DEFAULT_REPO_PATH)
    if not (path / "tsr" / "system.py").is_file():
        LOG.info("Cloning TripoSR into %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["git", "clone", "-q", TRIPOSR_GIT, str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "-q", TRIPOSR_COMMIT], check=False)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ReconstructionError(
                f"TripoSR source not found at {path} and cloning failed ({exc}). "
                f"Run scripts/setup_colab.sh or set UNVEIL_TRIPOSR_PATH."
            ) from exc
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    install_torchmcubes_shim()
    return path


class TripoSRReconstructor(Reconstructor):
    name = "triposr"
    # Matches TripoSR's canonical input camera (tsr.system.TSR.render defaults).
    camera = CameraSpec(distance=1.9, fov_deg=40.0)

    def __init__(self, device: str = "cuda", model_id: str = "stabilityai/TripoSR", chunk_size: int = 8192, repo_path: Optional[str] = None) -> None:
        self.device = device
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.repo_path = repo_path
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        ensure_triposr_importable(self.repo_path)
        LOG.info("Loading TripoSR %s", self.model_id)
        model = load_tsr(self.model_id)
        model.renderer.set_chunk_size(self.chunk_size)
        self._model = model.to(self.device).eval()

    def to(self, device: str) -> None:
        if self._model is not None:
            self._model.to(device)
        self.device = device

    def prepare_input(self, rgba: Image.Image, foreground_ratio: float) -> Image.Image:
        # TripoSR was trained on objects composited over 50% grey.
        return composite_on(resize_foreground(rgba, foreground_ratio), gray=0.5)

    def reconstruct(self, image: Image.Image, mc_resolution: int = 256, mc_threshold: float = 25.0) -> ReconstructionResult:
        import torch

        self.load()
        model = self._model
        with torch.no_grad():
            scene_codes = model([image.convert("RGB")], device=self.device)
            kwargs = {"resolution": mc_resolution, "threshold": mc_threshold}
            if "has_vertex_color" in inspect.signature(model.extract_mesh).parameters:
                kwargs["has_vertex_color"] = True
            try:
                native = model.extract_mesh(scene_codes, **kwargs)[0]
            except (RuntimeError, ValueError, IndexError) as exc:
                raise ReconstructionError(f"TripoSR mesh extraction failed: {exc}") from exc
        if len(native.faces) == 0:
            raise ReconstructionError("TripoSR produced an empty mesh (no surface at this threshold)")

        vertices = np.asarray(native.vertices, dtype=np.float64) @ NATIVE_TO_VIEW.T
        colors = None
        if native.visual.kind == "vertex":
            colors = np.asarray(native.visual.vertex_colors)
        mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(native.faces), vertex_colors=colors, process=False)
        return ReconstructionResult(mesh=mesh, handle=scene_codes[0].detach().to("cpu"), meta={"mc_resolution": mc_resolution})

    @property
    def supports_color_query(self) -> bool:
        return True

    def query_color(self, handle: Any, points: np.ndarray, chunk: int = 262_144) -> np.ndarray:
        import torch

        self.load()
        model = self._model
        native = np.asarray(points, dtype=np.float64) @ NATIVE_TO_VIEW  # inverse of a rotation = transpose
        scene_code = handle.to(self.device)
        out = np.empty((len(native), 3), np.float32)
        with torch.no_grad():
            for start in range(0, len(native), chunk):
                pts = torch.from_numpy(native[start : start + chunk]).float().to(self.device)
                rgb = model.renderer.query_triplane(model.decoder, pts, scene_code)["color"]
                out[start : start + chunk] = rgb.float().clamp(0, 1).cpu().numpy()
        return out
