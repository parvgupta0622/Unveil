"""Headless runner, for reproducing results without the UI.

    python -m unveil.cli --sketch examples/mug.png --prompt "a red ceramic mug"
    python -m unveil.cli --preload-only        # just download all model weights
"""
from __future__ import annotations

import argparse
import json
import sys

from PIL import Image

from .config import DEPTH_PRESETS, UnveilConfig
from .pipeline import PipelineError, RunOptions, RunResult, UnveilPipeline
from .preprocess import SketchError


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Unveil: sketch -> 3D (GLB/OBJ)")
    p.add_argument("--sketch", help="path to the sketch image")
    p.add_argument("--prompt", default="")
    p.add_argument("--negative", default="")
    p.add_argument("--mode", choices=["precise", "freeform"], default="precise")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variations", type=int, default=4, choices=[3, 4])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--top-k", type=int, default=2, choices=[1, 2])
    p.add_argument("--mc-resolution", type=int, default=256)
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--no-texture", action="store_true", help="skip UV texture baking (vertex colours only)")
    p.add_argument("--texture-resolution", type=int, default=1024)
    p.add_argument("--depth", default="1.0", help=f"depth scale or preset name {list(DEPTH_PRESETS)}")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--low-vram", action="store_true")
    p.add_argument("--preload-only", action="store_true", help="download and load every model, then exit")
    args = p.parse_args(argv)

    config = UnveilConfig(low_vram=True if args.low_vram else None)
    if args.output_dir:
        config.output_dir = args.output_dir
    pipe = UnveilPipeline(config)
    if args.preload_only:
        pipe.preload()
        print("All models downloaded and loaded.")
        return 0
    if not args.sketch:
        p.error("--sketch is required unless --preload-only is given")

    depth = DEPTH_PRESETS.get(args.depth.capitalize()) or float(args.depth)
    opts = RunOptions(
        prompt=args.prompt,
        negative_prompt=args.negative,
        control_mode=args.mode,
        seed=args.seed,
        num_variations=args.variations,
        steps=args.steps,
        top_k_images=args.top_k,
        mc_resolution=args.mc_resolution,
        smooth_iterations=args.smooth,
        bake_texture=not args.no_texture,
        texture_resolution=args.texture_resolution,
        depth_scale=depth,
    )
    try:
        for ev in pipe.run(Image.open(args.sketch), opts):
            if isinstance(ev, RunResult):
                print(json.dumps({"glb": str(ev.bundle.glb), "obj_zip": str(ev.bundle.obj_zip), "run_dir": str(ev.run_dir), "timings_s": ev.summary["timings_s"], "validation_passed": ev.bundle.report["passed"], "warnings": ev.warnings}, indent=2))
            elif ev.message:
                print(f"[{ev.stage}] {ev.message}", flush=True)
    except (SketchError, PipelineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
