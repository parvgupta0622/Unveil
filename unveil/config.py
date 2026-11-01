"""Central configuration. Every tunable used by the pipeline lives here so runs are reproducible."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CreativityLevel:
    """One rung of the ControlNet creativity ladder.

    Lower conditioning scale / earlier guidance end gives the diffusion model more freedom
    to deviate from the strokes; higher values follow the sketch more literally.
    """

    name: str
    conditioning_scale: float
    guidance_end: float


DEFAULT_LADDER: Tuple[CreativityLevel, ...] = (
    CreativityLevel("strict", 1.3, 1.0),
    CreativityLevel("balanced", 1.0, 0.9),
    CreativityLevel("creative", 0.8, 0.8),
    CreativityLevel("free", 0.6, 0.65),
)

# UI label -> internal mode. "Precise" = lineart ControlNet, "Freeform" = scribble ControlNet.
CONTROL_MODES = {
    "precise": "lllyasviel/control_v11p_sd15_lineart",
    "freeform": "lllyasviel/control_v11p_sd15_scribble",
}

DEPTH_PRESETS = {"Thin": 0.5, "Medium": 1.0, "Thick": 1.6}
DEPTH_RANGE = (0.3, 2.5)

# "auto" decides from the object category (prompt keywords, else CLIP zero-shot); see mesh/hollow.py.
INTERIOR_MODES = ("auto", "hollow", "solid")
WALL_RANGE = (0.02, 0.10)  # wall thickness as a fraction of the object's largest dimension


@dataclass
class GenerationConfig:
    base_model: str = "Lykon/dreamshaper-8"
    controlnets: dict = field(default_factory=lambda: dict(CONTROL_MODES))
    resolution: int = 512
    steps: int = 20
    guidance_scale: float = 7.0
    ladder: Tuple[CreativityLevel, ...] = DEFAULT_LADDER
    prompt_template: str = (
        "{subject}, a single complete object, centered, full view, isolated on a plain white "
        "background, product render, soft studio lighting, clean, highly detailed"
    )
    default_subject: str = "an object"
    negative_prompt: str = (
        "multiple objects, cluttered background, scenery, room, text, watermark, logo, cropped, "
        "out of frame, cut off, border, frame, cast shadow, lowres, blurry, noisy, deformed, "
        "extra parts, duplicate"
    )


@dataclass
class ScoringConfig:
    clip_model: str = "openai/clip-vit-base-patch32"
    w_iou: float = 0.45
    w_clip: float = 0.35
    w_clean: float = 0.20
    # Variations whose silhouette IoU with the sketch is below this are only used if nothing passes.
    iou_floor: float = 0.35


@dataclass
class ReconstructionConfig:
    backend: str = "triposr"
    model_id: str = "stabilityai/TripoSR"
    repo_path: Optional[str] = None  # defaults to third_party/TripoSR (auto-cloned if missing)
    chunk_size: int = 8192
    mc_resolution: int = 256
    mc_threshold: float = 25.0
    top_k_images: int = 2
    foreground_ratios: Tuple[float, ...] = (0.75, 0.85)


@dataclass
class MeshConfig:
    smooth_iterations: int = 5
    max_faces: int = 120_000  # decimate above this (keeps GLB size and UV-unwrap time sane)
    texture_resolution: int = 1024


@dataclass
class HollowConfig:
    wall_fraction: float = 0.05
    voxel_resolution: int = 160  # voxels along the object's largest dimension (TripoSR's own grid is ~170)
    max_outputs: int = 2  # final 3D models per build (one per selected variation)


@dataclass
class CadConfig:
    resolution: int = 192  # voxels along the part's largest dimension (CAD drawing -> 3D)
    max_faces: int = 40_000  # marching cubes over-tessellates flat faces; decimate down to this


@dataclass
class VerifyConfig:
    render_size: int = 256
    w_iou: float = 0.5
    w_edge: float = 0.3
    w_retention: float = 0.2
    edge_sigma: float = 0.03  # chamfer distance (fraction of canvas) at which edge score drops to 1/e
    min_faces: int = 500
    min_depth_ratio: float = 0.04  # depth extent / max(width, height); below this the mesh is a flat card


@dataclass
class UnveilConfig:
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    hollow: HollowConfig = field(default_factory=HollowConfig)
    cad: CadConfig = field(default_factory=CadConfig)
    rembg_model: str = "u2net"
    device: str = "auto"
    # None = decide from available VRAM (offload idle models to CPU on GPUs with < 13 GB).
    low_vram: Optional[bool] = None
    output_dir: str = field(default_factory=lambda: os.environ.get("UNVEIL_OUTPUT_DIR", str(REPO_ROOT / "outputs")))
