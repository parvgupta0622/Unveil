"""Gradio interface: sketch/upload input, live stage board, every pipeline step with previews, variation
choice, two interactive 3D previews, depth control and downloads in every format."""
from __future__ import annotations

import html
import inspect
import threading
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np

from .config import DEPTH_PRESETS, DEPTH_RANGE, REPO_ROOT, WALL_RANGE, UnveilConfig
from .mesh import format_report
from .pipeline import IMAGE_STAGES, STAGES, ImageRun, PipelineError, RunOptions, RunResult, StageEvent, UnveilPipeline
from .preprocess import SketchError
from .utils import LOG, image_from_any, to_jsonable

MODE_CHOICES = [("Precise", "precise"), ("Freeform", "freeform")]
INTERIOR_CHOICES = [("Auto-detect", "auto"), ("Hollow", "hollow"), ("Solid", "solid")]
CUSTOM = "Custom"
EXAMPLES = [
    ("mug.png", "a glossy red ceramic coffee mug"),
    ("chair.png", "a wooden dining chair"),
    ("lamp.png", "a modern table lamp with a white fabric shade and brass base"),
    ("bottle.png", ""),
]
# Babylon.js orbit camera (alpha, beta, radius): look at the model's front (+Z) slightly from above.
CAMERA_POSITION = (90, 80, None)

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="amber",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    block_border_width="0px",
    block_shadow="none",
    button_primary_background_fill="linear-gradient(90deg, *primary_600, *primary_500)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_700, *primary_600)",
    button_primary_text_color="white",
)

CSS = """
.gradio-container {max-width: 1360px !important; margin: auto}
#hero {background: linear-gradient(120deg, #0f2a33 0%, #0d5c5a 55%, #14857b 100%); border-radius: 18px;
       padding: 26px 30px; color: #f1f7f6; margin-bottom: 6px}
#hero * {color: #f1f7f6 !important}
#hero h1 {margin: 0; font-size: 2.1em; letter-spacing: -0.02em}
#hero p {margin: 6px 0 14px; opacity: 0.85; font-size: 1.02em}
#hero .flow {display: flex; flex-wrap: wrap; gap: 6px; align-items: center}
#hero .chip {background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18); border-radius: 999px;
             padding: 3px 11px; font-size: 0.85em; white-space: nowrap}
#hero .arrow {opacity: 0.55}
.card {background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
       border-radius: 16px !important; padding: 16px 18px !important; box-shadow: 0 1px 3px rgba(15, 42, 51, 0.06)}
.step-title h3 {margin: 0 0 2px !important; font-size: 1.12em}
.step-title p {margin: 0 0 6px !important; opacity: 0.75; font-size: 0.92em}
.badge {display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px; margin-right: 8px;
        border-radius: 50%; background: var(--color-accent); color: white; font-size: 0.85em; font-weight: 700}
.small-note, .small-note p {font-size: 0.88em; opacity: 0.8}
#stages .bar {height: 8px; border-radius: 99px; background: var(--neutral-200); overflow: hidden; margin: 6px 0 12px}
#stages .bar span {display: block; height: 100%; background: linear-gradient(90deg, #14857b, #2dd4bf); transition: width .4s}
#stages .headline {font-weight: 600}
#stages .stage {display: flex; gap: 10px; padding: 7px 0; border-bottom: 1px dashed var(--border-color-primary)}
#stages .stage:last-child {border-bottom: none}
#stages .dot {flex: none; width: 12px; height: 12px; margin-top: 5px; border-radius: 50%; background: var(--neutral-300)}
#stages .running .dot {background: #f59e0b; box-shadow: 0 0 0 4px rgba(245, 158, 11, .25); animation: pulse 1.2s infinite}
#stages .done .dot {background: #14b8a6}
#stages .error .dot {background: #ef4444}
#stages .pending .name {opacity: 0.55}
#stages .name {font-weight: 600}
#stages .time {float: right; font-weight: 400; opacity: 0.7; font-variant-numeric: tabular-nums}
#stages .note {font-size: 0.86em; opacity: 0.8}
#stages .warn {margin-top: 10px; padding: 8px 12px; border-radius: 10px; background: rgba(245, 158, 11, .12);
               border: 1px solid rgba(245, 158, 11, .35); font-size: 0.88em}
@keyframes pulse {50% {box-shadow: 0 0 0 7px rgba(245, 158, 11, 0)}}
.model-card {border-top: 4px solid var(--color-accent) !important}
.pick-row {align-items: end}
"""

HERO = """
<div id="hero">
  <h1>Unveil</h1>
  <p>Turn a rough sketch into a textured, downloadable 3D model.</p>
  <div class="flow">
    <span class="chip">✏️ Sketch</span><span class="arrow">→</span>
    <span class="chip">🧹 Preprocess</span><span class="arrow">→</span>
    <span class="chip">🎨 ControlNet × 4</span><span class="arrow">→</span>
    <span class="chip">🏆 Score &amp; pick</span><span class="arrow">→</span>
    <span class="chip">🧊 TripoSR × 2 framings</span><span class="arrow">→</span>
    <span class="chip">🫙 Hollow / solid</span><span class="arrow">→</span>
    <span class="chip">📦 GLB + OBJ</span>
  </div>
</div>
"""

SCORING_HELP = """
**How the "best" variation is chosen.** Every variation's background is removed (rembg), then it gets three scores:

| Score | Weight | What it measures |
|---|---|---|
| **Sketch IoU** | 45 % | Overlap between the generated object's mask and your sketch's filled silhouette: did it keep your shape? |
| **CLIP** | 35 % | CLIP ViT-B/32 similarity to your description (or to the sketch itself when there is no description), mapped to 0–1. |
| **Clean** | 20 % | Mean of four checks: object covers 8–70 % of the frame · one connected object · not touching the border · plain background. |

**Total = 0.45 · IoU + 0.35 · CLIP + 0.20 · Clean.** A variation with IoU below 0.35 fails the *structure check* and ranks below
every variation that passes, whatever its total. The two best (★ best, ☆ runner-up) are sent to TripoSR automatically.
Use the picker below to build the 3D models from any one or two variations instead.
"""

CANDIDATE_HELP = (
    "Each chosen variation is reconstructed twice (object filling 75 % and 85 % of the TripoSR frame). Every mesh is cleaned "
    "(largest connected part kept), re-projected into the input view and compared with your sketch: "
    "**0.5 · silhouette IoU + 0.3 · edge match + 0.2 · kept fraction**. Left: TripoSR input · right: blue = mesh, red = your "
    "sketch outline. ★ = the framing that became a final model."
)


def _kw(cls, **kwargs):
    """Drop kwargs this Gradio version does not accept, so the UI survives minor API changes."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return cls(**kwargs)
    return cls(**{k: v for k, v in kwargs.items() if k in params})


def _step(number: int, title: str, subtitle: str) -> gr.HTML:
    return gr.HTML(f"<div class='step-title'><h3><span class='badge'>{number}</span>{title}</h3><p>{subtitle}</p></div>")


class StageBoard:
    def __init__(self, skip=()) -> None:
        """skip: stages already finished in an earlier phase (e.g. the images, when only rebuilding 3D)."""
        self.state = {key: ("done" if key in skip else "pending") for key, _ in STAGES}
        self.notes = {key: ("from the image run" if key in skip else "") for key, _ in STAGES}
        self.started = {}
        self.elapsed = {}
        self.warnings = []
        self.headline = "Ready when you are — draw or upload a sketch."

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

    def finish(self, headline: str) -> None:
        self.headline = headline

    def fraction(self, stages=STAGES) -> float:
        return sum(self.state[k] == "done" for k, _ in stages) / len(stages)

    def html(self, stages=STAGES) -> str:
        rows = []
        for key, title in stages:
            st = self.state[key]
            t = f"{self.elapsed[key]:.1f}s" if key in self.elapsed else ""
            note = html.escape(self.notes[key]) if st != "pending" else ""
            rows.append(f"<div class='stage {st}'><span class='dot'></span><div style='flex:1'><div class='name'>{title}<span class='time'>{t}</span></div><div class='note'>{note}</div></div></div>")
        out = f"<div class='headline'>{html.escape(self.headline)}</div><div class='bar'><span style='width:{100 * self.fraction(stages):.0f}%'></span></div>" + "".join(rows)
        if self.elapsed and all(self.state[k] == "done" for k, _ in stages):
            out += f"<div class='note' style='margin-top:8px'><b>Total: {sum(self.elapsed.values()):.1f}s</b></div>"
        if self.warnings:
            out += "<div class='warn'>" + "<br>".join(f"⚠️ {html.escape(w)}" for w in self.warnings) + "</div>"
        return out


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


def _score_table(img: ImageRun) -> str:
    rows = sorted(img.score_rows(), key=lambda r: r["rank"])
    lines = ["| Rank | Variation | Conditioning · seed | Sketch IoU | CLIP | Clean | **Total** | Structure check |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['rank']} | #{r['index'] + 1} {r['level']} | {r['conditioning_scale']} · {r['seed']} | {r['iou']:.2f} | {r['clip']:.2f} | "
            f"{r['clean']:.2f} | **{r['total']:.2f}** | {'✓' if r['passes_floor'] else '✗ IoU < floor'} |"
        )
    return "\n".join(lines)


def _pick_choices(img: ImageRun):
    rank = {vi: r + 1 for r, vi in enumerate(img.ranking)}
    return [(f"#{v.index + 1} {v.level.name} (rank {rank[v.index]})", v.index) for v in img.variations]


def _model_header(o) -> str:
    return f"#### Model {o.slot + 1} · from variation #{o.variation + 1} ({o.level})\n<span class='small-note'>TripoSR candidate #{o.candidate + 1} · verification score {o.score.total:.2f}</span>"


def _interior_note(o) -> str:
    lines = [f"**Interior:** {o.interior.describe()}"]
    if o.hollow:
        lines.append(f"<span class='small-note'>{o.hollow.describe()}</span>")
    if o.interior.top_labels:
        lines.append("<span class='small-note'>CLIP top guesses: " + ", ".join(f"{label} {p:.2f}" for label, p in o.interior.top_labels) + "</span>")
    lines.append("<span class='small-note'>Cross-section: a cut through the middle, front to back.</span>")
    return "  \n".join(lines)


def build_demo(config: Optional[UnveilConfig] = None, preload: bool = False) -> gr.Blocks:
    config = config or UnveilConfig()
    state = {"pipe": None}
    lock = threading.Lock()
    n_slots = config.hollow.max_outputs

    def get_pipe() -> UnveilPipeline:
        with lock:
            if state["pipe"] is None:
                state["pipe"] = UnveilPipeline(config)
            return state["pipe"]

    if preload:
        get_pipe().preload()

    # ------------------------------------------------------------------ layout
    with gr.Blocks(title="Unveil — Sketch to 3D", theme=THEME, css=CSS) as demo:
        gr.HTML(HERO)
        input_source = gr.State("draw")
        run_state = gr.State(None)  # finished 3D build: {"run_id", "build_dir", "textured"}
        image_state = gr.State(None)  # image run whose variations the picker refers to

        with gr.Row(equal_height=False):
            with gr.Column(scale=6, elem_classes="card"):
                _step(1, "Your sketch", "Draw or upload a front view of one object, then describe it (optional).")
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
                            height=500,
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
                        info="Precise follows your lines closely. Freeform tolerates rough or open strokes.",
                    )
                    interior = gr.Radio(
                        INTERIOR_CHOICES,
                        value="auto",
                        label="Interior",
                        info="Auto hollows cups, bowls, bottles, vases… (from your description, else CLIP) and keeps the rest solid.",
                    )
                with gr.Row():
                    wall = gr.Slider(WALL_RANGE[0], WALL_RANGE[1], value=config.hollow.wall_fraction, step=0.005, label="Wall thickness (hollow objects, × object size)")
                    bake = gr.Checkbox(value=True, label="Bake UV texture (best colour in OBJ & other tools; ~10–30 s per model). Off = vertex colours.")
                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        seed = gr.Number(value=42, label="Seed", precision=0)
                        num_variations = gr.Slider(3, 4, value=4, step=1, label="ControlNet variations")
                    with gr.Row():
                        steps = gr.Slider(10, 40, value=20, step=1, label="Diffusion steps")
                        guidance = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="Prompt guidance (CFG)")
                    with gr.Row():
                        top_k = gr.Slider(1, n_slots, value=n_slots, step=1, label="Variations auto-sent to TripoSR (× 2 framings each)")
                        mc_res = gr.Dropdown([192, 256, 320], value=256, label="Marching-cubes resolution")
                    with gr.Row():
                        smooth = gr.Slider(0, 15, value=5, step=1, label="Smoothing iterations")
                        tex_res = gr.Dropdown([512, 1024, 2048], value=1024, label="Texture resolution")
                    negative = gr.Textbox(label="Extra negative prompt", placeholder="things to avoid")
                with gr.Row():
                    run_btn = gr.Button("✨ Generate 3D", variant="primary", size="lg", scale=3)
                    images_btn = gr.Button("🖼️ Images first — I'll pick", variant="secondary", size="lg", scale=2)
                if examples:
                    gr.Examples(examples=examples, inputs=[upload, prompt], label="Example sketches (load into the Upload tab)")

            with gr.Column(scale=4, elem_classes="card"):
                gr.HTML("<div class='step-title'><h3>Progress</h3><p>Every stage, live, with timings.</p></div>")
                status = gr.HTML(StageBoard().html(), elem_id="stages")
                with gr.Accordion("Run details (scores, timings, versions)", open=False):
                    details = gr.JSON(label="run.json")

        with gr.Column(elem_classes="card"):
            _step(2, "Sketch preprocessing", "Your input, the cleaned strokes, the ControlNet conditioning image and the filled silhouette used for scoring.")
            prep_gallery = _kw(gr.Gallery, columns=4, height="auto", object_fit="contain", show_label=False, preview=False)

        with gr.Column(elem_classes="card"):
            _step(3, "ControlNet variations", "Four creativity levels (strict → free), each scored against your sketch. Pick which ones become 3D models.")
            var_gallery = _kw(gr.Gallery, columns=4, height="auto", object_fit="contain", show_label=False, preview=False)
            score_table = gr.Markdown()
            with gr.Accordion("How is the best variation chosen?", open=False):
                gr.Markdown(SCORING_HELP)
            with gr.Row(elem_classes="pick-row"):
                pick = gr.CheckboxGroup(choices=[], label=f"Build 3D from these variations (1–{n_slots}; each becomes a final model)", scale=4)
                build_btn = gr.Button("🧊 Build 3D from selected", variant="primary", scale=1, interactive=False)

        with gr.Column(elem_classes="card"):
            _step(4, "TripoSR candidates & verification", "Two framings per chosen variation; each mesh is re-projected and checked against your sketch.")
            gr.Markdown(f"<span class='small-note'>{CANDIDATE_HELP}</span>")
            cand_gallery = _kw(gr.Gallery, columns=2, height="auto", object_fit="contain", show_label=False, preview=False)

        with gr.Column(elem_classes="card"):
            _step(5, "Final 3D models", "Drag to orbit, scroll to zoom. Depth and downloads apply to every model.")
            with gr.Row(equal_height=False):
                depth_preset = gr.Radio(list(DEPTH_PRESETS) + [CUSTOM], value="Medium", label="Depth (Z-axis thickness)", info="Rescales the actual geometry along the view axis and re-exports every file.")
                depth = gr.Slider(DEPTH_RANGE[0], DEPTH_RANGE[1], value=1.0, step=0.05, label="Depth scale")
            slots = []
            with gr.Row(equal_height=False):
                for i in range(n_slots):
                    with gr.Column(elem_classes="card model-card", visible=i == 0) as card:
                        header = gr.Markdown(f"#### Model {i + 1}\n<span class='small-note'>Not generated yet.</span>")
                        view = _kw(gr.Model3D, label=f"Model {i + 1}", clear_color=(0.94, 0.96, 0.96, 1.0), camera_position=CAMERA_POSITION, height=400)
                        with gr.Row(equal_height=False):
                            section = gr.Image(label="Cross-section", height=200, interactive=False, scale=1, min_width=150)
                            with gr.Column(scale=2):
                                info = gr.Markdown()
                        files = gr.File(label="Downloads: .glb · .obj zip (with .mtl + texture)", file_count="multiple", interactive=False)
                        with gr.Accordion("Export validation", open=False):
                            report = gr.Markdown()
                    slots.append({"card": card, "header": header, "view": view, "section": section, "info": info, "files": files, "report": report})
            with gr.Row(equal_height=False):
                all_zip = gr.File(label="📦 Everything — every model in every format (zip)", interactive=False, scale=3)
                bake_btn = gr.Button("Bake UV textures for these models", size="sm", visible=False, scale=1)

        slot_components = [c for s in slots for c in s.values()]

        # ------------------------------------------------------------------ helpers
        def read_opts(prompt_v, mode_v, interior_v, wall_v, bake_v, depth_v, seed_v, nvar_v, steps_v, cfg_v, topk_v, mc_v, smooth_v, tex_v, neg_v) -> RunOptions:
            return RunOptions(
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
                interior=interior_v or "auto",
                wall_thickness=float(wall_v),
            )

        def clear_models() -> dict:
            out = {all_zip: None, bake_btn: gr.update(visible=False), run_state: None, cand_gallery: []}
            for i, s in enumerate(slots):
                out.update({s["card"]: gr.update(visible=i == 0), s["header"]: f"#### Model {i + 1}\n<span class='small-note'>Not generated yet.</span>", s["view"]: None, s["section"]: None, s["info"]: "", s["files"]: None, s["report"]: ""})
            return out

        def show_bundles(bundles, zip_path) -> dict:
            out = {all_zip: str(zip_path)}
            for i, s in enumerate(slots):
                if i < len(bundles):
                    b = bundles[i]
                    out.update({s["view"]: str(b.glb), s["files"]: [str(b.glb), str(b.obj_zip)], s["report"]: format_report(b.report)})
            return out

        def show_result(result: RunResult, board: StageBoard) -> dict:
            board.warnings = result.warnings
            board.finish(f"Done — {len(result.outputs)} model(s) ready to download.")
            out = clear_models()
            out.update(show_bundles([o.bundle for o in result.outputs], result.all_zip))
            for i, s in enumerate(slots):
                if i < len(result.outputs):
                    o = result.outputs[i]
                    out.update({s["card"]: gr.update(visible=True), s["header"]: _model_header(o), s["section"]: o.section, s["info"]: _interior_note(o)})
                else:
                    out[s["card"]] = gr.update(visible=False)
            textured = all(o.bundle.textured for o in result.outputs)
            out.update({
                status: board.html(),
                cand_gallery: result.candidate_gallery,
                details: to_jsonable(result.summary),
                run_state: {"run_id": result.run_id, "build_dir": str(result.build_dir), "textured": textured},
                bake_btn: gr.update(visible=not textured),
                pick: gr.update(value=result.selection),
            })
            return out

        def show_images(img: ImageRun) -> dict:
            return {
                var_gallery: img.variation_gallery,
                score_table: _score_table(img),
                pick: gr.update(choices=_pick_choices(img), value=img.auto_selection(n_slots)),
                build_btn: gr.update(interactive=True),
                image_state: img.run_id,
            }

        def stream(events, board: StageBoard, stages=STAGES):
            """Turn pipeline events into UI updates; returns the final ImageRun/RunResult via StopIteration."""
            current = stages[0][0]
            try:
                for ev in events:
                    if not isinstance(ev, StageEvent):
                        return ev
                    current = ev.stage
                    board.update(ev)
                    out = {status: board.html(stages)}
                    if "prep_images" in ev.payload:
                        out[prep_gallery] = ev.payload["prep_images"]
                    if "variations" in ev.payload:
                        out[var_gallery] = ev.payload["variations"]
                    if "image_run" in ev.payload:
                        out.update(show_images(ev.payload["image_run"]))
                    if "candidates" in ev.payload:
                        out[cand_gallery] = ev.payload["candidates"]
                    yield out
            except (SketchError, PipelineError) as exc:
                board.fail(getattr(exc, "stage", current), str(exc))
                yield {status: board.html(stages)}
                raise gr.Error(str(exc)) from exc
            except Exception as exc:  # noqa: BLE001
                LOG.exception("Unexpected failure")
                board.fail(current, f"{type(exc).__name__}: {exc}")
                yield {status: board.html(stages)}
                raise gr.Error(f"Unexpected error: {exc}") from exc
            return None

        # ------------------------------------------------------------------ handlers
        def start(draw_v, upload_v, source, *settings, images_only: bool):
            sketch = _pick_input(draw_v, upload_v, source)
            if sketch is None:
                raise gr.Error("Draw something on the canvas or upload a sketch first.")
            opts = read_opts(*settings)
            stages = IMAGE_STAGES if images_only else STAGES
            board = StageBoard()
            board.headline = "Starting…"
            first = clear_models()
            first.update({status: board.html(stages), prep_gallery: [], var_gallery: [], score_table: "", details: None, pick: gr.update(choices=[], value=[]), build_btn: gr.update(interactive=False), image_state: None})
            yield first
            pipe = get_pipe()
            events = pipe.run_images(sketch, opts) if images_only else pipe.run(sketch, opts)
            final = yield from stream(events, board, stages)
            if isinstance(final, RunResult):
                yield show_result(final, board)
            elif isinstance(final, ImageRun):
                board.warnings = final.warnings
                board.finish("Images ready — choose up to two variations below and press “Build 3D from selected”.")
                yield {status: board.html(stages), details: to_jsonable({"run_id": final.run_id, "variations": final.score_rows(), "variation_ranking": final.ranking})}

        def on_generate(*args):
            yield from start(*args, images_only=False)

        def on_images(*args):
            yield from start(*args, images_only=True)

        def on_build(run_id, selection, *settings):
            if not run_id:
                raise gr.Error("Generate the images first.")
            if not selection:
                raise gr.Error("Tick at least one variation.")
            if len(selection) > n_slots:
                raise gr.Error(f"Pick at most {n_slots} variations.")
            opts = read_opts("", "precise", *settings)
            board = StageBoard(skip=[k for k, _ in IMAGE_STAGES])
            board.headline = "Building 3D from your selection…"
            first = clear_models()
            first[status] = board.html()
            yield first
            final = yield from stream(get_pipe().run_3d(run_id, [int(i) for i in selection], opts), board)
            if isinstance(final, RunResult):
                yield show_result(final, board)

        def on_depth(run, value):
            preset_name = next((k for k, v in DEPTH_PRESETS.items() if abs(v - value) < 1e-6), CUSTOM)
            if not run:
                return {depth_preset: preset_name}
            try:
                bundles, zip_path = get_pipe().reexport(Path(run["build_dir"]), float(value))
            except PipelineError as exc:
                raise gr.Error(str(exc)) from exc
            out = show_bundles(bundles, zip_path)
            out[depth_preset] = preset_name
            return out

        def on_bake(run, value, tex_v):
            if not run:
                raise gr.Error("Generate a model first.")
            try:
                bundles, zip_path, warns = get_pipe().bake_existing(Path(run["build_dir"]), float(value), int(tex_v))
            except PipelineError as exc:
                raise gr.Error(str(exc)) from exc
            for w in warns:
                gr.Warning(w)
            textured = all(b.textured for b in bundles)
            out = show_bundles(bundles, zip_path)
            out.update({run_state: dict(run, textured=textured), bake_btn: gr.update(visible=not textured)})
            return out

        settings = [prompt, mode, interior, wall, bake, depth, seed, num_variations, steps, guidance, top_k, mc_res, smooth, tex_res, negative]
        build_settings = [interior, wall, bake, depth, seed, num_variations, steps, guidance, top_k, mc_res, smooth, tex_res, negative]
        outputs = [status, prep_gallery, var_gallery, score_table, pick, build_btn, image_state, cand_gallery, details, run_state, bake_btn, all_zip] + slot_components
        run_btn.click(on_generate, [sketchpad, upload, input_source] + settings, outputs, concurrency_limit=1, show_progress="minimal")
        images_btn.click(on_images, [sketchpad, upload, input_source] + settings, outputs, concurrency_limit=1, show_progress="minimal")
        build_btn.click(on_build, [image_state, pick] + build_settings, outputs, concurrency_limit=1, show_progress="minimal")

        draw_tab.select(lambda: "draw", None, input_source)
        upload_tab.select(lambda: "upload", None, input_source)
        upload.change(lambda img: "upload" if img is not None else gr.update(), upload, input_source)
        clear_btn.click(lambda: None, None, sketchpad)

        model_outputs = [all_zip] + [s[k] for s in slots for k in ("view", "files", "report")]
        depth_preset.input(lambda p: gr.update(value=DEPTH_PRESETS[p]) if p in DEPTH_PRESETS else gr.update(), depth_preset, depth).then(
            on_depth, [run_state, depth], model_outputs + [depth_preset]
        )
        depth.release(on_depth, [run_state, depth], model_outputs + [depth_preset])
        bake_btn.click(on_bake, [run_state, depth, tex_res], model_outputs + [run_state, bake_btn])

    return demo
