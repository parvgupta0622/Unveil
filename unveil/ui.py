"""Gradio interface: sketch/upload input, live stage board, interactive 3D preview, depth control, downloads."""
from __future__ import annotations

import inspect
import threading
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np

from .config import DEPTH_PRESETS, DEPTH_RANGE, REPO_ROOT, UnveilConfig
from .mesh import format_report
from .pipeline import STAGES, PipelineError, RunOptions, RunResult, StageEvent, UnveilPipeline
from .preprocess import SketchError
from .utils import LOG, image_from_any, to_jsonable

MODE_CHOICES = [("Precise", "precise"), ("Freeform", "freeform")]
CUSTOM = "Custom"
EXAMPLES = [
    ("mug.png", "a glossy red ceramic coffee mug"),
    ("chair.png", "a wooden dining chair"),
    ("lamp.png", "a modern table lamp with a white fabric shade and brass base"),
    ("bottle.png", ""),
]
# Babylon.js orbit camera (alpha, beta, radius): look at the model's front (+Z) slightly from above.
CAMERA_POSITION = (90, 80, None)

CSS = """
#unveil-title {margin-bottom: 0}
#stage-board {min-height: 220px}
.small-note {font-size: 0.85em; opacity: 0.8}
"""


def _kw(cls, **kwargs):
    """Drop kwargs this Gradio version does not accept, so the UI survives minor API changes."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return cls(**kwargs)
    return cls(**{k: v for k, v in kwargs.items() if k in params})


class StageBoard:
    ICONS = {"pending": "⬜", "running": "🔄", "done": "✅", "error": "❌"}

    def __init__(self) -> None:
        self.state = {key: "pending" for key, _ in STAGES}
        self.notes = {key: "" for key, _ in STAGES}
        self.started = {}
        self.elapsed = {}
        self.warnings = []
        self.headline = "Starting…"

    def update(self, ev: StageEvent) -> None:
        if ev.status == "start":
            self.state[ev.stage] = "running"
            self.started[ev.stage] = time.perf_counter()
        elif ev.status == "done":
            self.state[ev.stage] = "done"
            self.elapsed[ev.stage] = time.perf_counter() - self.started.get(ev.stage, time.perf_counter())
        if ev.message:
            self.notes[ev.stage] = ev.message
            self.headline = ev.message

    def fail(self, stage: str, message: str) -> None:
        self.state[stage] = "error"
        self.notes[stage] = message
        self.headline = message

    def fraction(self) -> float:
        done = sum(s == "done" for s in self.state.values())
        return min(0.99, done / len(STAGES))

    def markdown(self) -> str:
        rows = []
        for key, title in STAGES:
            t = f" · {self.elapsed[key]:.1f}s" if key in self.elapsed else ""
            note = f" — <span class='small-note'>{self.notes[key]}</span>" if self.notes[key] and self.state[key] != "pending" else ""
            rows.append(f"{self.ICONS[self.state[key]]} **{title}**{t}{note}")
        if self.elapsed and all(s == "done" for s in self.state.values()):
            rows.append(f"\n**Total: {sum(self.elapsed.values()):.1f}s**")
        if self.warnings:
            rows.append("\n" + "\n".join(f"⚠️ {w}" for w in self.warnings))
        return "  \n".join(rows)


def _is_blank(img) -> bool:
    arr = np.asarray(img.convert("RGBA"))
    if arr[..., 3].max() == 0:
        return True
    visible = arr[..., 3] > 0
    return bool((arr[..., :3][visible] > 245).all())


def _pick_input(draw, upload, source: str):
    order = [upload, draw] if source == "upload" else [draw, upload]
    for value in order:
        img = image_from_any(value)
        if img is not None and not _is_blank(img):
            return img
    return None


def build_demo(config: Optional[UnveilConfig] = None, preload: bool = False) -> gr.Blocks:
    config = config or UnveilConfig()
    state = {"pipe": None}
    lock = threading.Lock()

    def get_pipe() -> UnveilPipeline:
        with lock:
            if state["pipe"] is None:
                state["pipe"] = UnveilPipeline(config)
            return state["pipe"]

    if preload:
        get_pipe().preload()

    # ------------------------------------------------------------------ layout
    with gr.Blocks(title="Unveil — Sketch to 3D", theme=gr.themes.Soft(primary_hue="indigo"), css=CSS) as demo:
        gr.Markdown("# Unveil\n### Sketch → ControlNet image → TripoSR 3D model", elem_id="unveil-title")
        gr.Markdown(
            "**1.** Draw on the canvas or upload a sketch · **2.** Optionally describe the object · **3.** Press **Generate 3D**. "
            "Unveil creates several ControlNet images at different creativity levels, keeps the one most faithful to your sketch, "
            "reconstructs multiple TripoSR candidates, verifies them against your sketch silhouette, and exports the best as GLB/OBJ."
        )
        input_source = gr.State("draw")
        run_state = gr.State(None)

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("✏️ Draw") as draw_tab:
                        sketchpad = _kw(
                            gr.Sketchpad,
                            label="Sketch (brush, eraser, undo in the toolbar)",
                            type="pil",
                            image_mode="RGBA",
                            canvas_size=(512, 512),
                            brush=_kw(gr.Brush, default_size=4, colors=["#000000"], color_mode="fixed"),
                            eraser=_kw(gr.Eraser, default_size=16),
                            layers=False,
                            height=520,
                        )
                        clear_btn = gr.Button("🗑️ Reset canvas", size="sm")
                    with gr.Tab("📁 Upload") as upload_tab:
                        upload = gr.Image(label="Sketch image (PNG/JPG; photos of paper sketches work)", type="pil", image_mode="RGBA", height=420)
                        example_dir = REPO_ROOT / "examples"
                        examples = [[str(example_dir / f), p] for f, p in EXAMPLES if (example_dir / f).exists()]

                prompt = gr.Textbox(label="Describe the object (optional)", placeholder="e.g. a matte black ceramic vase with gold rim", lines=2)
                with gr.Row():
                    mode = gr.Radio(
                        MODE_CHOICES,
                        value="precise",
                        label="Sketch interpretation",
                        info="Precise: follows your lines closely. Freeform: tolerates rough or open strokes and fills in more detail.",
                    )
                bake = gr.Checkbox(value=True, label="Bake UV texture (best colour in OBJ & other tools; adds ~10–30 s). Off = vertex colours.")
                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        seed = gr.Number(value=42, label="Seed", precision=0)
                        num_variations = gr.Slider(3, 4, value=4, step=1, label="ControlNet variations")
                    with gr.Row():
                        steps = gr.Slider(10, 40, value=20, step=1, label="Diffusion steps")
                        guidance = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="Prompt guidance (CFG)")
                    with gr.Row():
                        top_k = gr.Slider(1, 2, value=2, step=1, label="Images sent to TripoSR (× 2 framings each)")
                        mc_res = gr.Dropdown([192, 256, 320], value=256, label="Marching-cubes resolution")
                    with gr.Row():
                        smooth = gr.Slider(0, 15, value=5, step=1, label="Smoothing iterations")
                        tex_res = gr.Dropdown([512, 1024, 2048], value=1024, label="Texture resolution")
                    negative = gr.Textbox(label="Extra negative prompt", placeholder="things to avoid")
                run_btn = gr.Button("✨ Generate 3D", variant="primary", size="lg")
                if examples:
                    gr.Examples(examples=examples, inputs=[upload, prompt], label="Example sketches (load into the Upload tab)")

            with gr.Column(scale=7):
                status = gr.Markdown(StageBoard().markdown(), elem_id="stage-board")
                model3d = _kw(gr.Model3D, label="3D preview — drag to orbit, scroll to zoom", clear_color=(0.93, 0.94, 0.97, 1.0), camera_position=CAMERA_POSITION, height=460)
                with gr.Group():
                    with gr.Row():
                        depth_preset = gr.Radio(list(DEPTH_PRESETS) + [CUSTOM], value="Medium", label="Depth (Z-axis thickness)", info="Rescales the actual geometry along the view axis and re-exports the files.")
                        depth = gr.Slider(DEPTH_RANGE[0], DEPTH_RANGE[1], value=1.0, step=0.05, label="Depth scale")
                with gr.Row():
                    glb_file = gr.File(label="Download GLB")
                    obj_file = gr.File(label="Download OBJ (.zip with .mtl / texture)")
                bake_btn = gr.Button("Bake UV texture for this model", size="sm", visible=False)
                report = gr.Markdown()
                with gr.Tabs():
                    with gr.Tab("Sketch processing"):
                        prep_gallery = _kw(gr.Gallery, columns=4, height=260, object_fit="contain", show_label=False)
                    with gr.Tab("ControlNet variations"):
                        var_gallery = _kw(gr.Gallery, columns=4, height=300, object_fit="contain", show_label=False)
                    with gr.Tab("3D candidates"):
                        gr.Markdown("<span class='small-note'>Blue = mesh reprojected into the input view, red = your sketch outline. ★ = selected.</span>")
                        cand_gallery = _kw(gr.Gallery, columns=4, height=300, object_fit="contain", show_label=False)
                    with gr.Tab("Run details"):
                        details = gr.JSON(label="Scores, timings, versions")

        # ------------------------------------------------------------------ handlers
        def on_generate(draw_v, upload_v, source, prompt_v, mode_v, bake_v, depth_v, seed_v, nvar_v, steps_v, cfg_v, topk_v, mc_v, smooth_v, tex_v, neg_v, progress=gr.Progress()):
            sketch = _pick_input(draw_v, upload_v, source)
            if sketch is None:
                raise gr.Error("Draw something on the canvas or upload a sketch first.")
            opts = RunOptions(
                prompt=prompt_v or "",
                negative_prompt=neg_v or "",
                control_mode=mode_v or "precise",
                seed=int(seed_v if seed_v is not None else 42),
                num_variations=int(nvar_v),
                steps=int(steps_v),
                guidance_scale=float(cfg_v),
                top_k_images=int(topk_v),
                mc_resolution=int(mc_v),
                smooth_iterations=int(smooth_v),
                bake_texture=bool(bake_v),
                texture_resolution=int(tex_v),
                depth_scale=float(depth_v),
            )
            board = StageBoard()
            yield {status: board.markdown(), model3d: None, glb_file: None, obj_file: None, report: "", prep_gallery: [], var_gallery: [], cand_gallery: [], details: None, bake_btn: gr.update(visible=False)}
            current = STAGES[0][0]
            try:
                pipe = get_pipe()
                result = None
                for ev in pipe.run(sketch, opts):
                    if isinstance(ev, RunResult):
                        result = ev
                        break
                    current = ev.stage
                    board.update(ev)
                    progress(board.fraction(), desc=board.headline)
                    out = {status: board.markdown()}
                    if "prep_images" in ev.payload:
                        out[prep_gallery] = ev.payload["prep_images"]
                    if "variations" in ev.payload:
                        out[var_gallery] = ev.payload["variations"]
                    if "candidates" in ev.payload:
                        out[cand_gallery] = ev.payload["candidates"]
                    yield out
            except (SketchError, PipelineError) as exc:
                board.fail(getattr(exc, "stage", current), str(exc))
                yield {status: board.markdown()}
                raise gr.Error(str(exc)) from exc
            except Exception as exc:  # noqa: BLE001
                LOG.exception("Unexpected failure")
                board.fail(current, f"{type(exc).__name__}: {exc}")
                yield {status: board.markdown()}
                raise gr.Error(f"Unexpected error: {exc}") from exc

            board.warnings = result.warnings
            b = result.bundle
            yield {
                status: board.markdown(),
                model3d: str(b.glb),
                glb_file: str(b.glb),
                obj_file: str(b.obj_zip),
                report: format_report(b.report),
                details: to_jsonable(result.summary),
                run_state: {"run_id": result.run_id, "run_dir": str(result.run_dir), "textured": b.textured},
                bake_btn: gr.update(visible=not b.textured),
            }

        def on_depth(run, value):
            preset_name = next((k for k, v in DEPTH_PRESETS.items() if abs(v - value) < 1e-6), CUSTOM)
            if not run:
                return gr.update(), gr.update(), gr.update(), gr.update(), preset_name
            try:
                b = get_pipe().reexport(Path(run["run_dir"]), float(value))
            except PipelineError as exc:
                raise gr.Error(str(exc)) from exc
            return str(b.glb), str(b.glb), str(b.obj_zip), format_report(b.report), preset_name

        def on_bake(run, value, tex_v):
            if not run:
                raise gr.Error("Generate a model first.")
            try:
                b, warns = get_pipe().bake_existing(run["run_id"], Path(run["run_dir"]), float(value), int(tex_v))
            except PipelineError as exc:
                raise gr.Error(str(exc)) from exc
            for w in warns:
                gr.Warning(w)
            run = dict(run, textured=b.textured)
            return str(b.glb), str(b.glb), str(b.obj_zip), format_report(b.report), run, gr.update(visible=not b.textured)

        inputs = [sketchpad, upload, input_source, prompt, mode, bake, depth, seed, num_variations, steps, guidance, top_k, mc_res, smooth, tex_res, negative]
        outputs = [status, model3d, glb_file, obj_file, report, prep_gallery, var_gallery, cand_gallery, details, run_state, bake_btn]
        run_btn.click(on_generate, inputs, outputs, concurrency_limit=1)

        draw_tab.select(lambda: "draw", None, input_source)
        upload_tab.select(lambda: "upload", None, input_source)
        upload.change(lambda img: "upload" if img is not None else gr.update(), upload, input_source)
        clear_btn.click(lambda: None, None, sketchpad)

        depth_outputs = [model3d, glb_file, obj_file, report, depth_preset]
        depth_preset.input(lambda p: gr.update(value=DEPTH_PRESETS[p]) if p in DEPTH_PRESETS else gr.update(), depth_preset, depth).then(
            on_depth, [run_state, depth], depth_outputs
        )
        depth.release(on_depth, [run_state, depth], depth_outputs)
        bake_btn.click(on_bake, [run_state, depth, tex_res], [model3d, glb_file, obj_file, report, run_state, bake_btn])

    return demo
