"""Launch the Unveil Gradio app.

    python app.py --share            # Colab: prints a public *.gradio.live link
    python app.py --preload          # download/load all models before serving
"""
from __future__ import annotations

import argparse

from unveil.config import REPO_ROOT, UnveilConfig
from unveil.ui import build_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Unveil: sketch to 3D")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link (needed on Colab)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--preload", action="store_true", help="load all models at startup instead of on first run")
    parser.add_argument("--low-vram", action="store_true", help="offload idle models to CPU (GPUs with < 13 GB)")
    parser.add_argument("--output-dir", default=None, help="where runs are written (default: ./outputs)")
    args = parser.parse_args()

    config = UnveilConfig(low_vram=True if args.low_vram else None)
    if args.output_dir:
        config.output_dir = args.output_dir

    demo = build_demo(config, preload=args.preload)
    demo.queue(max_size=8).launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        allowed_paths=[config.output_dir, str(REPO_ROOT / "examples")],  # examples load from any working directory
        show_error=True,
    )


if __name__ == "__main__":
    main()
