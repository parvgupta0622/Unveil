"""Gradio interface: settings and sketch on top, then each pipeline step appears below as soon as it has results —
preprocessing, variations (with choice), TripoSR candidates and the final models in a 360° OBJ viewer with downloads."""
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
STAGE_SHORT = {"preprocess": "Preprocess", "generate": "ControlNet", "select_image": "Scoring", "reconstruct": "TripoSR", "interior": "Interior", "refine": "Refine", "export": "Export"}

# Palette: ink #1a535c · aqua #4ecdc4 · mint #f7fff7 · coral #ff6b6b · sun #ffe66d
INK = gr.themes.Color(
    c50="#effbfa", c100="#d6f5f2", c200="#aee9e4", c300="#82ddd5", c400="#4ecdc4", c500="#33b0a8",
    c600="#268f89", c700="#1f716f", c800="#1a535c", c900="#15434b", c950="#0e2f35", name="unveil_ink",
)
CORAL = gr.themes.Color(
    c50="#fff1f1", c100="#ffe0e0", c200="#ffc4c4", c300="#ff9d9d", c400="#ff8383", c500="#ff6b6b",
    c600="#ed4f4f", c700="#c83c3c", c800="#a33434", c900="#862f2f", c950="#491414", name="unveil_coral",
)

THEME = gr.themes.Soft(
    primary_hue=INK,
    secondary_hue=CORAL,
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f7fff7",
    body_background_fill_dark="#0e2f35",
    block_background_fill="white",
    block_background_fill_dark="#15434b",
    block_border_width="0px",
    block_shadow="none",
    color_accent="#4ecdc4",
    color_accent_soft="#d6f5f2",
    slider_color="#4ecdc4",
    checkbox_background_color_selected="#1a535c",
    button_primary_background_fill="#1a535c",
    button_primary_background_fill_hover="#226a75",
    button_primary_text_color="#f7fff7",
    button_primary_background_fill_dark="#4ecdc4",
    button_primary_background_fill_hover_dark="#82ddd5",
    button_primary_text_color_dark="#0e2f35",
    button_secondary_background_fill="white",
    button_secondary_background_fill_hover="#d6f5f2",
    button_secondary_border_color="#1a535c",
    button_secondary_text_color="#1a535c",
    button_secondary_background_fill_dark="#15434b",
    button_secondary_border_color_dark="#4ecdc4",
    button_secondary_text_color_dark="#4ecdc4",
)

CSS = """
.gradio-container {max-width: 1120px !important; margin: auto; overflow-x: clip !important; overflow-y: visible !important}  /* hidden would break sticky */
#hero {background: radial-gradient(circle at 85% 20%, rgba(78,205,196,.45) 0%, rgba(78,205,196,0) 45%), #1a535c;
       border-radius: 20px; padding: 26px 30px; margin-bottom: 4px}
#hero * {color: #f7fff7 !important}
#hero h1 {margin: 0; font-size: 2.2em; letter-spacing: -0.02em; display: inline-block; border-bottom: 4px solid #ffe66d; line-height: 1.15}
#hero p {margin: 10px 0 14px; opacity: 0.9; font-size: 1.03em}
#hero .flow {display: flex; flex-wrap: wrap; gap: 6px; align-items: center}
#hero .chip {background: rgba(247,255,247,0.10); border: 1px solid rgba(78,205,196,0.55); border-radius: 999px;
             padding: 3px 11px; font-size: 0.85em; white-space: nowrap}
#hero .arrow {opacity: 0.55}
.card {background: var(--block-background-fill); border: 1px solid rgba(78,205,196,.35);
       border-radius: 18px !important; padding: 18px 20px !important; box-shadow: 0 2px 10px rgba(26,83,92,.06)}
.step-title h3 {margin: 0 0 2px !important; font-size: 1.15em; display: flex; align-items: center}
.step-title p {margin: 0 0 8px !important; opacity: 0.75; font-size: 0.92em}
.badge {display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; margin-right: 10px; flex: none;
        border-radius: 50%; background: #4ecdc4; color: #1a535c; font-size: 0.85em; font-weight: 800}
.small-note, .small-note p {font-size: 0.88em; opacity: 0.8}
.reveal {scroll-margin-top: 150px; animation: rise .35s ease-out}
@keyframes rise {from {opacity: 0; transform: translateY(10px)} to {opacity: 1; transform: none}}
#progress-card:has(.sb.running) {position: sticky; top: 8px; z-index: 50}
.sb .sb-head {display: flex; justify-content: space-between; gap: 12px; font-weight: 600}
.sb .total {font-variant-numeric: tabular-nums; opacity: .8; white-space: nowrap}
.sb .bar {height: 8px; border-radius: 99px; background: rgba(78,205,196,.18); overflow: hidden; margin: 8px 0 10px}
.sb .bar span {display: block; height: 100%; background: linear-gradient(90deg, #4ecdc4, #1a535c); transition: width .4s}
.sb .stages {display: flex; flex-wrap: wrap; gap: 6px}
.sb .stage {display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px; font-size: 0.84em;
            border: 1px solid var(--border-color-primary); cursor: default}
.sb .dot {width: 9px; height: 9px; border-radius: 50%; background: var(--neutral-300); flex: none}
.sb .pending {opacity: 0.5}
.sb .running {background: rgba(255,230,109,.35); border-color: #e9cc3f}
.sb .running .dot {background: #ffe66d; box-shadow: 0 0 0 3px rgba(233,204,63,.45); animation: pulse 1.2s infinite}
.sb .done {background: rgba(78,205,196,.14); border-color: rgba(78,205,196,.6)}
.sb .done .dot {background: #4ecdc4}
.sb .error {background: rgba(255,107,107,.12); border-color: #ff6b6b}
.sb .error .dot {background: #ff6b6b}
.sb .time {opacity: 0.65; font-variant-numeric: tabular-nums}
.sb .warn {margin-top: 10px; padding: 8px 12px; border-radius: 10px; background: rgba(255,230,109,.28);
           border: 1px solid #e9cc3f; font-size: 0.88em}
@keyframes pulse {50% {box-shadow: 0 0 0 6px rgba(233,204,63,0)}}
.model-card {border-top: 5px solid #4ecdc4 !important}
.pick-row {align-items: end}
unveil-viewer {display: block}
.uv-stage {position: relative; height: 440px; border-radius: 14px; overflow: hidden; cursor: grab; touch-action: none;
           background: radial-gradient(circle at 50% 35%, #ffffff 0%, #e3f7f4 75%)}
.uv-stage:active {cursor: grabbing}
.dark .uv-stage {background: radial-gradient(circle at 50% 35%, #226a75 0%, #0e2f35 75%)}
.uv-msg {position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; padding: 20px; text-align: center; opacity: .75}
.uv-msg.err {color: #c83c3c; opacity: 1}
.uv-bar {display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 8px}
.uv-bar button {border: 1px solid #1a535c; color: #1a535c; background: transparent; border-radius: 999px; padding: 3px 12px;
                font-size: .85em; cursor: pointer}
.uv-bar button.on {background: #1a535c; color: #f7fff7}
.dark .uv-bar button {border-color: #4ecdc4; color: #4ecdc4}
.dark .uv-bar button.on {background: #4ecdc4; color: #0e2f35}
.uv-hint {font-size: .8em; opacity: .65; margin-left: auto}
#err-overlay {position: fixed !important; inset: 0; z-index: 1000; margin: 0 !important; padding: 16px !important;
              max-width: none !important; justify-content: center; align-items: center; background: rgba(14,47,53,.55)}
#err-card {flex: none !important; width: min(520px, 100%); background: var(--block-background-fill); border-radius: 16px !important;
           border-top: 6px solid #ff6b6b; padding: 18px 22px !important; box-shadow: 0 24px 60px rgba(0,0,0,.3)}
#err-card h3 {margin-top: 0 !important}
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

# Runs once on page load: the <unveil-viewer> element (three.js OBJ viewer, free 360° rotation) and auto-scroll to new sections.
PAGE_JS = """
() => {
  const CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0";
  let libs = null;
  const loadLibs = () => libs || (libs = Promise.all([
    import(`${CDN}/+esm`),
    import(`${CDN}/examples/jsm/loaders/OBJLoader.js/+esm`),
    import(`${CDN}/examples/jsm/loaders/MTLLoader.js/+esm`),
    import(`${CDN}/examples/jsm/controls/TrackballControls.js/+esm`),
  ]).then(([THREE, o, m, t]) => ({THREE, OBJLoader: o.OBJLoader, MTLLoader: m.MTLLoader, TrackballControls: t.TrackballControls})));
  const fileUrl = (path) => {
    const root = ((window.gradio_config && window.gradio_config.root) || window.location.origin).replace(/\\/$/, "");
    return `${root}/gradio_api/file=${path.split("/").map(encodeURIComponent).join("/")}`;
  };

  class UnveilViewer extends HTMLElement {
    connectedCallback() {
      if (this.started) return;
      this.started = true;
      this.alive = true;
      this.innerHTML = `<div class="uv-stage"><div class="uv-msg">Loading 3D preview…</div></div>
        <div class="uv-bar"><button type="button" data-a="spin" class="on">⟳ Auto-spin</button>
        <button type="button" data-a="reset">⌖ Reset view</button><button type="button" data-a="wire">▦ Wireframe</button>
        <span class="uv-hint">Drag to turn it any way · click, then scroll to zoom · right-drag to pan</span></div>`;
      loadLibs().then((L) => this.init(L)).catch((e) => this.fail("Could not load the 3D preview. The files below still download.", e));
    }
    disconnectedCallback() {
      this.alive = false;
      cancelAnimationFrame(this.raf);
      if (this.resizer) this.resizer.disconnect();
      if (this.controls) this.controls.dispose();
      if (this.renderer) { this.renderer.dispose(); this.renderer.forceContextLoss(); }
    }
    fail(message, err) {
      if (err) console.error(err);
      const m = this.querySelector(".uv-msg");
      if (m) { m.textContent = message; m.classList.add("err"); }
    }
    setSpin(on) {
      this.spin = on;
      const b = this.querySelector('[data-a="spin"]');
      if (b) b.classList.toggle("on", on);
    }
    async init({THREE, OBJLoader, MTLLoader, TrackballControls}) {
      const url = fileUrl(this.dataset.obj);
      const dir = url.slice(0, url.lastIndexOf("/") + 1);
      const res = await fetch(url, {cache: "no-store"});
      if (!res.ok) return this.fail(`Could not load the OBJ file (HTTP ${res.status}).`);
      const text = await res.text();
      const loader = new OBJLoader();
      const mtlName = (text.match(/^mtllib\\s+(.+?)\\s*$/m) || [])[1];
      if (mtlName) {
        try {
          const materials = await new MTLLoader().setPath(dir).loadAsync(mtlName);
          materials.preload();
          loader.setMaterials(materials);
        } catch (e) { console.warn("Unveil viewer: material file not loaded", e); }
      }
      if (!this.alive) return;
      const model = loader.parse(text);
      model.traverse((o) => {
        if (!o.isMesh) return;
        for (const m of [].concat(o.material)) {
          m.side = THREE.DoubleSide;
          if (o.geometry.attributes.color) m.vertexColors = true;
          if ("shininess" in m) m.shininess = 12;
        }
      });
      const box = new THREE.Box3().setFromObject(model);
      model.position.sub(box.getCenter(new THREE.Vector3()));
      const pivot = new THREE.Group();
      pivot.add(model);
      const radius = box.getSize(new THREE.Vector3()).length() / 2 || 1;

      const stage = this.querySelector(".uv-stage");
      const renderer = this.renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      stage.innerHTML = "";
      stage.appendChild(renderer.domElement);
      const scene = new THREE.Scene();
      scene.add(pivot);
      scene.add(new THREE.HemisphereLight(0xf7fff7, 0x1a535c, 2.0));
      const camera = new THREE.PerspectiveCamera(35, 1, radius / 100, radius * 100);
      const key = new THREE.DirectionalLight(0xffffff, 1.8);
      key.position.set(1, 1.5, 2);
      camera.add(key);  // the key light follows the viewer, so whichever side you turn to is lit
      scene.add(camera);
      const dist = radius / Math.sin(THREE.MathUtils.degToRad(17.5)) * 1.05;
      camera.position.set(0, dist * 0.26, dist * 0.97);
      camera.lookAt(0, 0, 0);

      const controls = this.controls = new TrackballControls(camera, renderer.domElement);
      controls.rotateSpeed = 3.0;
      controls.zoomSpeed = 1.2;
      controls.panSpeed = 0.8;
      controls.dynamicDampingFactor = 0.15;
      controls.minDistance = radius * 0.6;
      controls.maxDistance = radius * 20;
      controls.addEventListener("start", () => this.setSpin(false));
      // The wheel scrolls the page until you click or drag in the viewer; then it zooms (until the pointer leaves).
      let engaged = false;
      stage.addEventListener("pointerdown", () => { engaged = true; }, true);
      stage.addEventListener("pointerleave", () => { engaged = false; });
      stage.addEventListener("wheel", (e) => { if (!engaged) e.stopPropagation(); }, true);

      const resize = () => {
        const w = stage.clientWidth || 400, h = stage.clientHeight || 440;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        controls.handleResize();
      };
      resize();
      this.resizer = new ResizeObserver(resize);
      this.resizer.observe(stage);

      let wire = false;
      this.querySelector(".uv-bar").addEventListener("click", (e) => {
        const b = e.target.closest("button");
        if (!b) return;
        if (b.dataset.a === "spin") this.setSpin(!this.spin);
        if (b.dataset.a === "reset") { controls.reset(); pivot.rotation.set(0, 0, 0); }
        if (b.dataset.a === "wire") {
          wire = !wire;
          b.classList.toggle("on", wire);
          model.traverse((o) => { if (o.isMesh) for (const m of [].concat(o.material)) m.wireframe = wire; });
        }
      });
      this.setSpin(true);
      const tick = () => {
        if (!this.alive) return;
        this.raf = requestAnimationFrame(tick);
        if (this.spin) pivot.rotation.y += 0.008;
        controls.update();
        renderer.render(scene, camera);
      };
      tick();
    }
  }
  if (!customElements.get("unveil-viewer")) customElements.define("unveil-viewer", UnveilViewer);

  // Auto-scroll: when a result section appears, bring it into view (unless the user is scrolling around themselves).
  let lastUserScroll = 0;
  for (const ev of ["wheel", "touchmove", "keydown"]) window.addEventListener(ev, () => { lastUserScroll = Date.now(); }, {passive: true});
  const shown = new Set();
  let queued = false;
  const check = () => {
    queued = false;
    let target = null;
    for (const el of document.querySelectorAll(".reveal")) {
      const visible = el.getClientRects().length > 0;
      if (visible && !shown.has(el)) { shown.add(el); target = target || el; }
      if (!visible) shown.delete(el);
    }
    if (!target || Date.now() - lastUserScroll < 2500) return;
    const r = target.getBoundingClientRect();
    if (r.top > window.innerHeight * 0.6 || r.top < 0) target.scrollIntoView({behavior: "smooth", block: "start"});
  };
  new MutationObserver(() => { if (!queued) { queued = true; requestAnimationFrame(check); } })
    .observe(document.body, {subtree: true, childList: true, attributes: true, attributeFilter: ["class", "style"]});
}
"""


class InputError(ValueError):
    """Something the user has to fix before a run can start (shown in the error popup, not as a crash)."""


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

    def fail(self, stage: str) -> None:
        """Mark where it stopped; the message itself goes in the error popup only."""
        self.state[stage] = "error"
        for key, value in self.state.items():
            if value == "running":
                self.state[key] = "pending"
        self.headline = f"Stopped at “{dict(STAGES).get(stage, stage)}”."

    def finish(self, headline: str) -> None:
        self.headline = headline

    def fraction(self, stages=STAGES) -> float:
        return sum(self.state[k] == "done" for k, _ in stages) / len(stages)

    def html(self, stages=STAGES) -> str:
        states = [self.state[k] for k, _ in stages]
        running = "running" in states or ("error" not in states and 0 < self.fraction(stages) < 1)  # also between two stages
        total = f"{sum(self.elapsed.values()):.1f} s" if self.elapsed else ""
        pills = []
        for key, title in stages:
            st = self.state[key]
            t = f"<span class='time'>{self.elapsed[key]:.1f}s</span>" if key in self.elapsed else ""
            tip = html.escape(f"{title}: {self.notes[key]}" if self.notes[key] else title, quote=True)
            pills.append(f"<span class='stage {st}' title=\"{tip}\"><span class='dot'></span>{STAGE_SHORT.get(key, title)}{t}</span>")
        out = (
            f"<div class='sb{' running' if running else ''}'><div class='sb-head'><span>{html.escape(self.headline)}</span><span class='total'>{total}</span></div>"
            f"<div class='bar'><span style='width:{100 * self.fraction(stages):.0f}%'></span></div><div class='stages'>{''.join(pills)}</div>"
        )
        if self.warnings:
            out += "<div class='warn'>" + "<br>".join(f"⚠️ {html.escape(w)}" for w in self.warnings) + "</div>"
        return out + "</div>"


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


def _viewer(bundle) -> str:
    """The OBJ (with its .mtl/texture next to it) in the in-page three.js viewer; the stamp forces a reload after re-exports."""
    return f"<unveil-viewer data-obj=\"{html.escape(str(bundle.obj), quote=True)}\" data-v=\"{time.time_ns()}\"></unveil-viewer>"


def _error_text(exc: Exception) -> str:
    if isinstance(exc, InputError):
        title, message, hint = "Not quite ready", str(exc), ""
    elif isinstance(exc, SketchError):
        title, message, hint = "Couldn't read the sketch", str(exc), "Try darker, thicker strokes, or switch to Freeform."
    elif isinstance(exc, PipelineError):
        stage = dict(STAGES).get(exc.stage, exc.stage)
        title, message, hint = f"{stage} failed", str(exc), "Your sketch and settings are kept — adjust and try again."
    else:
        title, message, hint = "Unexpected error", f"{type(exc).__name__}: {exc}", "The full traceback is in the Colab / terminal log."
    return f"### ⚠️ {title}\n\n{message}" + (f"\n\n<span class='small-note'>{hint}</span>" if hint else "")


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

    # ------------------------------------------------------------------ layout (one column: inputs on top, results appear below)
    with gr.Blocks(title="Unveil — Sketch to 3D", theme=THEME, css=CSS, js=PAGE_JS) as demo:
        gr.HTML(HERO)
        input_source = gr.State("draw")
        run_state = gr.State(None)  # finished 3D build: {"run_id", "build_dir", "textured"}
        image_state = gr.State(None)  # image run whose variations the picker refers to

        with gr.Column(elem_classes="card"):
            _step(1, "Settings", "How your sketch is read and what the model should look like. The defaults work well.")
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
                    steps = gr.Slider(10, 40, value=20, step=1, label="Diffusion steps")
                    guidance = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="Prompt guidance (CFG)")
                with gr.Row():
                    top_k = gr.Slider(1, n_slots, value=n_slots, step=1, label="Variations auto-sent to TripoSR (× 2 framings each)")
                    mc_res = gr.Dropdown([192, 256, 320], value=256, label="Marching-cubes resolution")
                    smooth = gr.Slider(0, 15, value=5, step=1, label="Smoothing iterations")
                    tex_res = gr.Dropdown([512, 1024, 2048], value=1024, label="Texture resolution")
                negative = gr.Textbox(label="Extra negative prompt", placeholder="things to avoid")

        with gr.Column(elem_classes="card"):
            _step(2, "Your sketch", "Draw or upload a front view of one object, describe it (optional), then generate.")
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
                run_btn = gr.Button("✨ Generate 3D", variant="primary", size="lg", scale=3)
                images_btn = gr.Button("🖼️ Images first — I'll pick", variant="secondary", size="lg", scale=2)
            if examples:
                gr.Examples(examples=examples, inputs=[upload, prompt], label="Example sketches (load into the Upload tab)")

        # Everything below starts hidden and appears as soon as it has something to show.
        with gr.Column(elem_classes="card reveal", elem_id="progress-card", visible=False) as progress_card:
            status = gr.HTML(StageBoard().html())

        with gr.Column(elem_classes="card reveal", visible=False) as prep_card:
            _step(3, "Sketch preprocessing", "Your input, the cleaned strokes, the ControlNet conditioning image and the filled silhouette used for scoring.")
            prep_gallery = _kw(gr.Gallery, columns=4, height="auto", object_fit="contain", show_label=False, preview=False)

        with gr.Column(elem_classes="card reveal", visible=False) as var_card:
            _step(4, "ControlNet variations", "Four creativity levels (strict → free), each scored against your sketch. Pick which ones become 3D models.")
            var_gallery = _kw(gr.Gallery, columns=4, height="auto", object_fit="contain", show_label=False, preview=False)
            with gr.Column(visible=False) as pick_box:
                score_table = gr.Markdown()
                with gr.Accordion("How is the best variation chosen?", open=False):
                    gr.Markdown(SCORING_HELP)
                with gr.Row(elem_classes="pick-row"):
                    pick = gr.CheckboxGroup(choices=[], label=f"Build 3D from these variations (1–{n_slots}; each becomes a final model)", scale=4)
                    build_btn = gr.Button("🧊 Build 3D from selected", variant="primary", scale=1)

        with gr.Column(elem_classes="card reveal", visible=False) as cand_card:
            _step(5, "TripoSR candidates & verification", "Two framings per chosen variation; each mesh is re-projected and checked against your sketch.")
            gr.Markdown(f"<span class='small-note'>{CANDIDATE_HELP}</span>")
            cand_gallery = _kw(gr.Gallery, columns=2, height="auto", object_fit="contain", show_label=False, preview=False)

        with gr.Column(elem_classes="card reveal", visible=False) as final_card:
            _step(6, "Final 3D models", "The exported OBJ, live: drag to turn it in any direction, click then scroll to zoom. Depth and downloads apply to every model.")
            with gr.Row(equal_height=False):
                depth_preset = gr.Radio(list(DEPTH_PRESETS) + [CUSTOM], value="Medium", label="Depth (Z-axis thickness)", info="Rescales the actual geometry along the view axis and re-exports every file.")
                depth = gr.Slider(DEPTH_RANGE[0], DEPTH_RANGE[1], value=1.0, step=0.05, label="Depth scale")
            final_note = gr.Markdown(visible=False)
            slots = []
            for i in range(n_slots):
                with gr.Column(elem_classes="card model-card", visible=False) as card:
                    header = gr.Markdown()
                    view = gr.HTML()
                    with gr.Row(equal_height=False):
                        section = gr.Image(label="Cross-section", height=200, interactive=False, scale=1, min_width=160)
                        with gr.Column(scale=3):
                            info = gr.Markdown()
                            files = gr.File(label="Downloads: .glb · .obj zip (with .mtl + texture)", file_count="multiple", interactive=False)
                    with gr.Accordion("Export validation", open=False):
                        report = gr.Markdown()
                slots.append({"card": card, "header": header, "view": view, "section": section, "info": info, "files": files, "report": report})
            with gr.Row(equal_height=False):
                all_zip = gr.File(label="📦 Everything — every model in every format (zip)", interactive=False, scale=3)
                bake_btn = gr.Button("Bake UV textures for these models", size="sm", visible=False, scale=1)

        with gr.Column(elem_classes="card", visible=False) as details_card:
            with gr.Accordion("Run details (scores, timings, versions)", open=False):
                details = gr.JSON(label="run.json")

        with gr.Column(visible=False, elem_id="err-overlay") as err_box:
            with gr.Column(elem_id="err-card"):
                err_md = gr.Markdown()
                err_close = gr.Button("OK", variant="primary")

        slot_components = [c for s in slots for c in s.values()]
        error_outputs = [err_box, err_md]

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

        def popup(exc: Exception) -> dict:
            return {err_box: gr.update(visible=True), err_md: _error_text(exc)}

        def guarded(handler):
            """Any failure becomes the one error popup, instead of a toast plus an error badge on every output."""
            if inspect.isgeneratorfunction(handler):
                def run(*args):
                    try:
                        yield from handler(*args)
                    except Exception as exc:  # noqa: BLE001
                        if not isinstance(exc, (InputError, SketchError, PipelineError)):
                            LOG.exception("Unexpected failure")
                        yield popup(exc)
            else:
                def run(*args):
                    try:
                        return handler(*args)
                    except Exception as exc:  # noqa: BLE001
                        if not isinstance(exc, (InputError, SketchError, PipelineError)):
                            LOG.exception("Unexpected failure")
                        return popup(exc)
            return run

        def clear_models() -> dict:
            out = {all_zip: None, bake_btn: gr.update(visible=False), run_state: None, cand_gallery: [], final_note: gr.update(value="", visible=False),
                   cand_card: gr.update(visible=False), final_card: gr.update(visible=False), err_box: gr.update(visible=False)}
            for s in slots:
                out.update({s["card"]: gr.update(visible=False), s["header"]: "", s["view"]: "", s["section"]: None, s["info"]: "", s["files"]: None, s["report"]: ""})
            return out

        def show_bundles(bundles, zip_path) -> dict:
            out = {all_zip: str(zip_path)}
            for i, s in enumerate(slots):
                if i < len(bundles):
                    b = bundles[i]
                    out.update({s["view"]: _viewer(b), s["files"]: [str(b.glb), str(b.obj_zip)], s["report"]: format_report(b.report)})
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
            textured = all(o.bundle.textured for o in result.outputs)
            out.update({
                status: board.html(),
                cand_card: gr.update(visible=True),
                cand_gallery: result.candidate_gallery,
                final_card: gr.update(visible=True),
                details_card: gr.update(visible=True),
                details: to_jsonable(result.summary),
                run_state: {"run_id": result.run_id, "build_dir": str(result.build_dir), "textured": textured},
                bake_btn: gr.update(visible=not textured),
                pick: gr.update(value=result.selection),
            })
            return out

        def show_images(img: ImageRun) -> dict:
            return {
                var_card: gr.update(visible=True),
                var_gallery: img.variation_gallery,
                pick_box: gr.update(visible=True),
                score_table: _score_table(img),
                pick: gr.update(choices=_pick_choices(img), value=img.auto_selection(n_slots)),
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
                        out.update({prep_card: gr.update(visible=True), prep_gallery: ev.payload["prep_images"]})
                    if "variations" in ev.payload:
                        out.update({var_card: gr.update(visible=True), var_gallery: ev.payload["variations"]})
                    if "image_run" in ev.payload:
                        out.update(show_images(ev.payload["image_run"]))
                    if "candidates" in ev.payload:
                        out.update({cand_card: gr.update(visible=True), cand_gallery: ev.payload["candidates"]})
                    yield out
            except Exception as exc:
                board.fail(getattr(exc, "stage", current))
                yield {status: board.html(stages)}
                raise
            return None

        # ------------------------------------------------------------------ handlers
        def start(draw_v, upload_v, source, *settings, images_only: bool):
            sketch = _pick_input(draw_v, upload_v, source)
            if sketch is None:
                raise InputError("Draw something on the canvas or upload a sketch first.")
            opts = read_opts(*settings)
            stages = IMAGE_STAGES if images_only else STAGES
            board = StageBoard()
            board.headline = "Starting…"
            first = clear_models()
            first.update({
                status: board.html(stages), progress_card: gr.update(visible=True), prep_card: gr.update(visible=False), prep_gallery: [],
                var_card: gr.update(visible=False), var_gallery: [], pick_box: gr.update(visible=False), score_table: "",
                details_card: gr.update(visible=False), details: None, pick: gr.update(choices=[], value=[]), image_state: None,
            })
            yield first
            pipe = get_pipe()
            events = pipe.run_images(sketch, opts) if images_only else pipe.run(sketch, opts)
            final = yield from stream(events, board, stages)
            if isinstance(final, RunResult):
                yield show_result(final, board)
            elif isinstance(final, ImageRun):
                board.warnings = final.warnings
                board.finish("Images ready — choose up to two variations and press “Build 3D from selected”.")
                yield {status: board.html(stages), details_card: gr.update(visible=True),
                       details: to_jsonable({"run_id": final.run_id, "variations": final.score_rows(), "variation_ranking": final.ranking})}

        def on_generate(*args):
            yield from start(*args, images_only=False)

        def on_images(*args):
            yield from start(*args, images_only=True)

        def on_build(run_id, selection, *settings):
            if not run_id:
                raise InputError("Generate the images first.")
            if not selection:
                raise InputError("Tick at least one variation.")
            if len(selection) > n_slots:
                raise InputError(f"Pick at most {n_slots} variations.")
            opts = read_opts("", "precise", *settings)
            board = StageBoard(skip=[k for k, _ in IMAGE_STAGES])
            board.headline = "Building 3D from your selection…"
            first = clear_models()
            first.update({status: board.html(), progress_card: gr.update(visible=True)})
            yield first
            final = yield from stream(get_pipe().run_3d(run_id, [int(i) for i in selection], opts), board)
            if isinstance(final, RunResult):
                yield show_result(final, board)

        def on_depth(run, value):
            preset_name = next((k for k, v in DEPTH_PRESETS.items() if abs(v - value) < 1e-6), CUSTOM)
            if not run:
                return {depth_preset: preset_name}
            bundles, zip_path = get_pipe().reexport(Path(run["build_dir"]), float(value))
            out = show_bundles(bundles, zip_path)
            out[depth_preset] = preset_name
            return out

        def on_bake(run, value, tex_v):
            if not run:
                raise InputError("Generate a model first.")
            bundles, zip_path, warns = get_pipe().bake_existing(Path(run["build_dir"]), float(value), int(tex_v))
            textured = all(b.textured for b in bundles)
            out = show_bundles(bundles, zip_path)
            out.update({
                run_state: dict(run, textured=textured),
                bake_btn: gr.update(visible=not textured),
                final_note: gr.update(value="\n".join(f"⚠️ {w}" for w in warns), visible=bool(warns)),
            })
            return out

        settings = [prompt, mode, interior, wall, bake, depth, seed, num_variations, steps, guidance, top_k, mc_res, smooth, tex_res, negative]
        build_settings = [interior, wall, bake, depth, seed, num_variations, steps, guidance, top_k, mc_res, smooth, tex_res, negative]
        sections = [progress_card, prep_card, var_card, pick_box, cand_card, final_card, details_card]
        outputs = [status, prep_gallery, var_gallery, score_table, pick, image_state, cand_gallery, details, run_state, bake_btn, all_zip, final_note] + sections + slot_components + error_outputs
        run_btn.click(guarded(on_generate), [sketchpad, upload, input_source] + settings, outputs, concurrency_limit=1, show_progress="hidden")
        images_btn.click(guarded(on_images), [sketchpad, upload, input_source] + settings, outputs, concurrency_limit=1, show_progress="hidden")
        build_btn.click(guarded(on_build), [image_state, pick] + build_settings, outputs, concurrency_limit=1, show_progress="hidden")
        err_close.click(lambda: gr.update(visible=False), None, err_box, show_progress="hidden")

        draw_tab.select(lambda: "draw", None, input_source)
        upload_tab.select(lambda: "upload", None, input_source)
        upload.change(lambda img: "upload" if img is not None else gr.update(), upload, input_source)
        clear_btn.click(lambda: None, None, sketchpad)

        model_outputs = [all_zip] + [s[k] for s in slots for k in ("view", "files", "report")] + error_outputs
        depth_preset.input(lambda p: gr.update(value=DEPTH_PRESETS[p]) if p in DEPTH_PRESETS else gr.update(), depth_preset, depth).then(
            guarded(on_depth), [run_state, depth], model_outputs + [depth_preset], show_progress="minimal"
        )
        depth.release(guarded(on_depth), [run_state, depth], model_outputs + [depth_preset], show_progress="minimal")
        bake_btn.click(guarded(on_bake), [run_state, depth, tex_res], model_outputs + [run_state, bake_btn, final_note], show_progress="minimal")

    return demo
