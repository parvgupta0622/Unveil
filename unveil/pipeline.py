"""End-to-end orchestration.

`UnveilPipeline.run()` is a generator: it yields `StageEvent`s while it works (the UI turns them into a
live progress board) and finally yields a `RunResult`. Every model is lazily loaded on first use.

A run has two phases that can also be driven separately, so users can pick the ControlNet variations:
`run_images()` (preprocess, generate, score -> `ImageRun`) and `run_3d()` (TripoSR candidates, hollow/solid
interior, refine, export -> `RunResult`, one final model per selected variation).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import trimesh
from PIL import Image

from . import cad
from .background import BackgroundRemover
from .config import UnveilConfig
from .generation import SketchImageGenerator, Variation, build_prompt
from .mesh import (
    CandidateScore,
    CleanStats,
    ConfidenceField,
    ConfidenceStats,
    ExportBundle,
    GuideStats,
    HollowStats,
    InteriorDecision,
    MeshAsset,
    MeshError,
    SymmetryStats,
    apply_guides,
    bake_texture,
    clean_mesh,
    decide_interior,
    decimate,
    export_asset,
    export_heatmap,
    hollow_out,
    mirror_front_back,
    save_confidence,
    score_candidate,
    section_image,
    see_through_holes,
    smooth_mesh,
    surface_confidence,
    zip_bundles,
)
from .preprocess import SketchError, SketchPrep, preprocess_sketch
from .reconstruction import ReconstructionError, create_reconstructor
from .scoring import ClipScorer, VariationScore, rank_variations, score_variations
from .utils import LOG, StageTimer, flatten_alpha, free_gpu_memory, gpu_total_gb, new_run_id, resolve_device, write_json

STAGES: List[Tuple[str, str]] = [
    ("preprocess", "Sketch preprocessing"),
    ("generate", "ControlNet variations"),
    ("select_image", "Variation scoring & selection"),
    ("reconstruct", "TripoSR candidates & verification"),
    ("interior", "Guides, symmetry & hollow / solid interior"),
    ("refine", "Mesh refinement & texture"),
    ("export", "Export & validation"),
]
IMAGE_STAGES = STAGES[:3]
# CAD drawing -> 3D (no neural models involved).
CAD_STAGES: List[Tuple[str, str]] = [
    ("cad_read", "Read the drawing"),
    ("cad_views", "Find front / top / side views"),
    ("cad_solid", "Build the solid"),
    ("export", "Export & validation"),
]


class PipelineError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass
class RunOptions:
    prompt: str = ""
    negative_prompt: str = ""
    control_mode: str = "precise"  # "precise" (lineart) | "freeform" (scribble)
    seed: int = 42
    num_variations: int = 4
    steps: int = 20
    guidance_scale: float = 7.0
    top_k_images: int = 2
    foreground_ratios: Tuple[float, ...] = (0.75, 0.85)
    mc_resolution: int = 256
    smooth_iterations: int = 5
    bake_texture: bool = True
    texture_resolution: int = 1024
    depth_scale: float = 1.0
    interior: str = "auto"  # auto | hollow | solid
    wall_thickness: float = 0.05  # fraction of the object's largest dimension
    front_back_symmetric: bool = False  # rebuild the hidden back as a mirror image of the visible front
    open_holes: bool = True  # drill gaps that are see-through in the image and drawn as a loop in the sketch


@dataclass
class GuideViews:
    """Optional extra sketches that trim TripoSR's guess (see mesh/guide.py)."""

    side: Optional[Image.Image] = None  # the object seen from its side
    top: Optional[Image.Image] = None  # seen from above, its front at the bottom of the sheet
    side_front: str = "left"  # which way the object's front faces in the side sketch

    @property
    def given(self) -> List[str]:
        return [name for name in ("side", "top") if getattr(self, name) is not None]


@dataclass
class StageEvent:
    stage: str
    status: str  # start | progress | done | warn
    message: str = ""
    fraction: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    index: int
    variation: int
    foreground_ratio: float
    mesh: Optional[trimesh.Trimesh] = None
    handle: Any = None
    stats: Optional[CleanStats] = None
    score: Optional[CandidateScore] = None
    overlay: Optional[Image.Image] = None
    input_image: Optional[Image.Image] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.mesh is not None and self.score is not None

    def summary(self) -> dict:
        out = {"index": self.index, "variation": self.variation, "foreground_ratio": self.foreground_ratio, "error": self.error}
        if self.stats:
            out["cleanup"] = asdict(self.stats)
        if self.score:
            out["score"] = asdict(self.score)
        return out


@dataclass
class ImageRun:
    """Result of the image phase: everything the 3D phase needs, kept in memory so users can re-pick."""

    run_id: str
    run_dir: Path
    opts: RunOptions
    prompt: str
    prep: SketchPrep
    prep_images: List[Tuple[Image.Image, str]]
    variations: List[Variation]
    rgbas: List[Image.Image]
    scores: List[VariationScore]
    ranking: List[int]
    variation_gallery: List[Tuple[Image.Image, str]]
    warnings: List[str]
    timings: Dict[str, float]

    def auto_selection(self, k: int) -> List[int]:
        return self.ranking[: max(1, min(k, len(self.ranking)))]

    def score_rows(self) -> List[Dict[str, Any]]:
        rank = {vi: r + 1 for r, vi in enumerate(self.ranking)}
        return [
            dict(asdict(s), level=self.variations[s.index].level.name, conditioning_scale=self.variations[s.index].level.conditioning_scale, seed=self.variations[s.index].seed, rank=rank[s.index])
            for s in self.scores
        ]


@dataclass
class FinalOutput:
    """One finished 3D model: the best verified TripoSR candidate for one selected variation."""

    slot: int
    variation: int
    level: str
    candidate: int
    asset_path: Path
    bundle: ExportBundle
    interior: InteriorDecision
    hollow: Optional[HollowStats]
    section: Image.Image
    score: CandidateScore
    symmetry: Optional[SymmetryStats] = None
    guide: Optional[GuideStats] = None
    confidence: Optional[ConfidenceStats] = None

    def summary(self) -> dict:
        return {
            "model": self.slot + 1,
            "variation": self.variation,
            "level": self.level,
            "candidate": self.candidate,
            "files": {"glb": str(self.bundle.glb), "obj_zip": str(self.bundle.obj_zip)},
            "interior": asdict(self.interior),
            "hollow": asdict(self.hollow) if self.hollow else None,
            "symmetry": asdict(self.symmetry) if self.symmetry else None,
            "guide": asdict(self.guide) if self.guide else None,
            "confidence": asdict(self.confidence) if self.confidence else None,
            "validation_passed": self.bundle.report.get("passed"),
        }


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    build_dir: Path
    outputs: List[FinalOutput]
    all_zip: Path
    selection: List[int]
    prep_images: List[Tuple[Image.Image, str]]
    variation_gallery: List[Tuple[Image.Image, str]]
    candidate_gallery: List[Tuple[Image.Image, str]]
    summary: Dict[str, Any]
    warnings: List[str]

    @property
    def bundle(self) -> ExportBundle:
        """The first (highest-ranked selected) model."""
        return self.outputs[0].bundle


@dataclass
class CadRun:
    """A model built from a three-view CAD drawing."""

    run_id: str
    build_dir: Path
    source: str  # uploaded file name
    model: "cad.CadModel"
    asset_path: Path
    bundle: ExportBundle
    all_zip: Path
    section: Image.Image
    summary: Dict[str, Any]
    warnings: List[str]


def select_best(candidates: List[Candidate]) -> Optional[Candidate]:
    usable = [c for c in candidates if c.ok]
    if not usable:
        return None
    valid = [c for c in usable if c.score.valid]
    return max(valid or usable, key=lambda c: c.score.total)


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def _versions() -> Dict[str, str]:
    out = {}
    for pkg in ("torch", "diffusers", "transformers", "trimesh", "gradio", "rembg", "xatlas"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


class UnveilPipeline:
    def __init__(self, config: Optional[UnveilConfig] = None) -> None:
        import torch

        self.cfg = config or UnveilConfig()
        self.device = resolve_device(self.cfg.device)
        self.dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        vram = gpu_total_gb(self.device)
        self.low_vram = self.cfg.low_vram if self.cfg.low_vram is not None else (self.device == "cuda" and vram < 13)
        LOG.info("Device %s (%.1f GB VRAM), dtype %s, low_vram=%s", self.device, vram, self.dtype, self.low_vram)

        rc = self.cfg.reconstruction
        self.generator = SketchImageGenerator(self.cfg.generation, self.device, self.dtype)
        self.clip = ClipScorer(self.cfg.scoring.clip_model, self.device, self.dtype)
        self.background = BackgroundRemover(self.cfg.rembg_model)
        self.reconstructor = create_reconstructor(rc.backend, device=self.device, model_id=rc.model_id, chunk_size=rc.chunk_size, repo_path=rc.repo_path)
        self._handles: "OrderedDict[str, Any]" = OrderedDict()  # asset path -> (colour handle, remap)
        self._image_runs: "OrderedDict[str, ImageRun]" = OrderedDict()

    # ------------------------------------------------------------------ model residency
    def _use(self, *active: str) -> None:
        """Make `active` models resident on the GPU; in low-VRAM mode park the rest on the CPU."""
        models = {"generator": self.generator, "clip": self.clip, "reconstructor": self.reconstructor}
        if self.low_vram:
            moved = False
            for name, model in models.items():
                if name not in active and model.loaded and model.device != "cpu":
                    model.to("cpu")
                    moved = True
            if moved:
                free_gpu_memory()
        for name in active:
            models[name].to(self.device)

    def preload(self, control_modes: Tuple[str, ...] = ("precise", "freeform")) -> None:
        """Download and initialise every model up front (avoids a slow first click)."""
        self._use("generator")
        for mode in control_modes:
            self.generator.load(mode)
        self._use("clip")
        self.clip.load()
        self.background.load()
        self._use("reconstructor")
        self.reconstructor.load()

    def _remember(self, cache: "OrderedDict[str, Any]", key: str, value: Any, limit: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    # ------------------------------------------------------------------ entry points
    def run(self, sketch: Image.Image, opts: RunOptions, guides: Optional[GuideViews] = None) -> Iterator[Union[StageEvent, RunResult]]:
        """Full automatic run: images, then 3D from the top-ranked `opts.top_k_images` variations."""
        yield from self._guard(self._run(sketch, opts, guides), "preprocess")

    def run_images(self, sketch: Image.Image, opts: RunOptions) -> Iterator[Union[StageEvent, ImageRun]]:
        """Image phase only; finishes with an `ImageRun` whose variations the user can choose from."""
        yield from self._guard(self._images(sketch, opts), "preprocess")

    def run_3d(self, run_id: str, selection: List[int], opts: Optional[RunOptions] = None, guides: Optional[GuideViews] = None) -> Iterator[Union[StageEvent, RunResult]]:
        """3D phase for user-chosen variation indices of an earlier `run_images()`/`run()`."""
        img = self._image_runs.get(run_id)
        if img is None:
            raise PipelineError("reconstruct", "This run's images are no longer in memory - generate the images again.")
        if opts is not None:  # the 3D settings may change between builds; the image settings may not
            keep = ("prompt", "negative_prompt", "control_mode", "seed", "num_variations", "steps", "guidance_scale")
            opts = replace(opts, **{k: getattr(img.opts, k) for k in keep})
        yield from self._guard(self._models(img, selection, opts or img.opts, "user", guides), "reconstruct")

    def run_cad(self, path: Path, depth_scale: float = 1.0) -> Iterator[Union[StageEvent, CadRun]]:
        """Solid from a DXF/DWG with front, top and side views (see unveil/cad.py)."""
        yield from self._guard(self._cad(Path(path), depth_scale), "cad_read")

    def _guard(self, events: Iterator[Any], stage: str) -> Iterator[Any]:
        try:
            for event in events:
                if isinstance(event, StageEvent):
                    stage = event.stage
                yield event
        except (SketchError, PipelineError):
            raise
        except cad.CadError as exc:
            raise PipelineError(stage, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - convert to a stage-tagged, user-readable error
            if _is_oom(exc):
                free_gpu_memory()
                raise PipelineError(stage, "GPU ran out of memory. Try fewer variations, a lower marching-cubes resolution, or launch with --low-vram.") from exc
            LOG.exception("Stage %s failed", stage)
            raise PipelineError(stage, f"{type(exc).__name__}: {exc}") from exc

    def _run(self, sketch: Image.Image, opts: RunOptions, guides: Optional[GuideViews] = None) -> Iterator[Union[StageEvent, RunResult]]:
        img = None
        for event in self._images(sketch, opts):
            if isinstance(event, ImageRun):
                img = event
            else:
                yield event
        yield from self._models(img, img.auto_selection(opts.top_k_images), opts, "auto", guides)

    # ------------------------------------------------------------------ phase 1: images
    def _images(self, sketch: Image.Image, opts: RunOptions) -> Iterator[Union[StageEvent, ImageRun]]:
        gen_cfg = self.cfg.generation
        warnings: List[str] = []
        timer = StageTimer()
        run_id = new_run_id()
        run_dir = Path(self.cfg.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Preprocess -----------------------------------------------------------------
        yield StageEvent("preprocess", "start", "Cleaning strokes and extracting the silhouette")
        with timer("preprocess"):
            source = flatten_alpha(sketch)
            prep = preprocess_sketch(sketch, size=gen_cfg.resolution)
            control = prep.control_image(opts.control_mode)
        prep_images = [
            (source, "Input"),
            (prep.clean_image(), "Cleaned strokes" + (" (filled shape detected)" if prep.filled_input else "")),
            (control, f"ControlNet conditioning ({opts.control_mode})"),
            (prep.silhouette_image(), "Sketch silhouette (used for scoring)"),
        ]
        for i, (img, _) in enumerate(prep_images):
            img.save(run_dir / f"prep_{i}.png")
        yield StageEvent("preprocess", "done", "Strokes cleaned, silhouette extracted", payload={"prep_images": prep_images})

        # 2. ControlNet variations ------------------------------------------------------
        levels = gen_cfg.ladder[: max(1, min(opts.num_variations, len(gen_cfg.ladder)))]
        prompt = build_prompt(gen_cfg, opts.prompt)
        negative = ", ".join(p for p in (gen_cfg.negative_prompt, opts.negative_prompt.strip()) if p)
        yield StageEvent("generate", "start", f"Loading models and generating {len(levels)} variations")
        variations: List[Variation] = []
        with timer("generate"):
            self._use("generator")
            for i, level in enumerate(levels):
                seed = int(opts.seed) + i
                yield StageEvent("generate", "progress", f"Variation {i + 1}/{len(levels)} · {level.name} (conditioning {level.conditioning_scale}, seed {seed})", fraction=i / len(levels))
                image = self.generator.generate(control, prompt, negative, level, seed, opts.control_mode, opts.steps, opts.guidance_scale)
                image.save(run_dir / f"variation_{i}_{level.name}.png")
                variations.append(Variation(i, level, seed, image))
                yield StageEvent("generate", "progress", payload={"variations": [(v.image, f"#{v.index + 1} {v.level.name} · seed {v.seed}") for v in variations]})
        yield StageEvent("generate", "done", f"{len(variations)} images generated", payload={"variations": [(v.image, f"#{v.index + 1} {v.level.name} · seed {v.seed}") for v in variations]})

        # 3. Score variations: sketch IoU + CLIP + clean background --------------------------
        yield StageEvent("select_image", "start", "Removing backgrounds and scoring variations")
        with timer("select_image"):
            rgbas = [self.background.remove(v.image) for v in variations]
            alphas = [np.asarray(r)[..., 3] for r in rgbas]
            self._use("clip")
            clip_scores = self.clip.similarity([v.image for v in variations], opts.prompt, prep.clean_image())
            scores = score_variations([v.image for v in variations], alphas, prep.silhouette, clip_scores, self.cfg.scoring)
            ranking = rank_variations(scores)
        if not scores[ranking[0]].passes_floor:
            warnings.append(f"No variation reached sketch IoU ≥ {self.cfg.scoring.iou_floor}; using the best available. Try 'Precise' mode or a clearer outline.")
        top = ranking[: max(1, min(opts.top_k_images, len(ranking)))]
        rank = {vi: r + 1 for r, vi in enumerate(ranking)}
        variation_gallery = []
        for v, s in zip(variations, scores):
            mark = "★ " if v.index == top[0] else ("☆ " if v.index in top else "")
            variation_gallery.append((v.image, f"{mark}#{v.index + 1} {v.level.name} · rank {rank[v.index]} · total {s.total:.2f} · IoU {s.iou:.2f} · CLIP {s.clip:.2f} · clean {s.clean:.2f}"))
        img_run = ImageRun(run_id, run_dir, opts, prompt, prep, prep_images, variations, rgbas, scores, ranking, variation_gallery, warnings, dict(timer.timings))
        self._remember(self._image_runs, run_id, img_run, 4)
        write_json({"run_id": run_id, "prompt_used": prompt, "options": asdict(opts), "variations": img_run.score_rows(), "variation_ranking": ranking}, run_dir / "images.json")
        yield StageEvent("select_image", "done", f"Best: #{top[0] + 1} {variations[top[0]].level.name}", payload={"variations": variation_gallery, "image_run": img_run})
        yield img_run

    # ------------------------------------------------------------------ phase 2: 3D
    def _models(self, img: ImageRun, selection: List[int], opts: RunOptions, selected_by: str, guides: Optional[GuideViews] = None) -> Iterator[Union[StageEvent, RunResult]]:
        selection = list(dict.fromkeys(int(i) for i in selection))
        limit = self.cfg.hollow.max_outputs
        if not selection or any(not 0 <= i < len(img.variations) for i in selection):
            raise PipelineError("reconstruct", f"Choose between 1 and {limit} of the {len(img.variations)} variations.")
        if len(selection) > limit:
            raise PipelineError("reconstruct", f"Choose at most {limit} variations (each one becomes a final 3D model).")
        variations, prep = img.variations, img.prep
        guides = guides or GuideViews()
        guide_masks = _guide_masks(guides)  # read the guide sketches first: a bad one fails before TripoSR runs
        warnings = list(img.warnings)
        timer = StageTimer()
        build_dir = img.run_dir / ("3d_v" + "-".join(str(i + 1) for i in selection))
        build_dir.mkdir(parents=True, exist_ok=True)

        # 4. TripoSR candidates -> cleanup -> verification -------------------------------
        jobs = [(vi, ratio) for vi in selection for ratio in opts.foreground_ratios]
        yield StageEvent("reconstruct", "start", f"Reconstructing {len(jobs)} candidates with TripoSR")
        candidates: List[Candidate] = []
        with timer("reconstruct"):
            self._use("reconstructor")
            for ci, (vi, ratio) in enumerate(jobs):
                yield StageEvent("reconstruct", "progress", f"Candidate {ci + 1}/{len(jobs)} · variation #{vi + 1} {variations[vi].level.name}, foreground {ratio:.2f}", fraction=ci / len(jobs))
                cand = Candidate(ci, vi, ratio)
                try:
                    cand.input_image = self.reconstructor.prepare_input(img.rgbas[vi], ratio)
                    result = self.reconstructor.reconstruct(cand.input_image, opts.mc_resolution, self.cfg.reconstruction.mc_threshold)
                    mesh, cand.stats = clean_mesh(result.mesh)
                    cand.mesh = smooth_mesh(mesh, opts.smooth_iterations)
                    cand.handle = result.handle
                    cand.score, cand.overlay = score_candidate(cand.mesh, prep.silhouette, self.reconstructor.camera, cand.stats, self.cfg.verify)
                except (ReconstructionError, MeshError, ValueError) as exc:
                    cand.error = str(exc)
                    warnings.append(f"Candidate {ci + 1} failed: {exc}")
                candidates.append(cand)
        finals = []
        for vi in selection:  # one final model per selected variation: its best verified framing
            best = select_best([c for c in candidates if c.variation == vi])
            if best is None:
                warnings.append(f"Every TripoSR candidate for variation #{vi + 1} failed; it produced no model.")
                continue
            if not best.score.valid:
                warnings.append(f"Variation #{vi + 1}: no candidate passed validity checks ({best.score.reason}); using the highest-scoring one.")
            finals.append(best)
        if not finals:
            raise PipelineError("reconstruct", "All TripoSR candidates failed. Try a sketch with a clearer closed outline.")
        final_ids = [f.index for f in finals]
        candidate_gallery = []
        for c in candidates:
            if c.ok:
                mark = f"★ model {final_ids.index(c.index) + 1} · " if c.index in final_ids else ""
                s = c.score
                caption = f"{mark}#{c.index + 1} variation #{c.variation + 1} {variations[c.variation].level.name} @ {c.foreground_ratio:.2f} · total {s.total:.2f} · IoU {s.iou:.2f} · edge {s.edge:.2f} · kept {s.retention:.0%}"
                if not s.valid:
                    caption += f" · invalid: {s.reason}"
                candidate_gallery.append((_side_by_side(c.input_image, c.overlay), caption))
                c.overlay.save(build_dir / f"candidate_{c.index}_overlay.png")
            else:
                candidate_gallery.append((Image.new("RGB", (512, 256), (245, 245, 245)), f"#{c.index + 1} failed: {c.error}"))
        yield StageEvent("reconstruct", "done", "Best: " + ", ".join(f"candidate #{f.index + 1}" for f in finals), payload={"candidates": candidate_gallery})

        # 5. Shape: guide views / see-through holes trim TripoSR's guess, an optional front-back mirror, then
        #    vessels get a real cavity. The confidence map is measured on the trimmed mesh, before mirroring.
        first = "Fitting your guide views" if guide_masks else ("Mirroring the front onto the back" if opts.front_back_symmetric else "Deciding hollow vs solid")
        yield StageEvent("interior", "start", first)
        shaped = []  # (candidate, mesh, decision, hollow stats, colour-query remap)
        symmetry: Dict[int, SymmetryStats] = {}
        guided: Dict[int, GuideStats] = {}
        confidence: Dict[int, ConfidenceField] = {}
        overlays = 0
        with timer("interior"):
            for slot, c in enumerate(finals):
                mesh, stats, remap, pinned = c.mesh, None, None, None
                holes = see_through_holes(np.asarray(img.rgbas[c.variation])[..., 3], prep.ink) if opts.open_holes else None
                if holes is not None and not holes[0].any():
                    holes = None
                if guide_masks or holes is not None:
                    yield StageEvent("interior", "progress", f"Model {slot + 1}: " + ("trimming to your guide views" if guide_masks else "opening see-through holes"), fraction=slot / len(finals))
                    fitted, reason = apply_guides(
                        mesh, self.reconstructor.camera, holes, guide_masks.get("side"), guide_masks.get("top"), guides.side_front != "right",
                        self.cfg.guide.resolution, self.cfg.guide.tolerance, min(opts.smooth_iterations, 3),
                    )
                    if fitted is None:
                        if reason:
                            warnings.append(f"Model {slot + 1}: guide views not applied ({reason}).")
                    else:
                        mesh, pinned, guided[slot] = fitted.mesh, fitted.pinned, fitted.stats
                        candidate_gallery += [(im, f"Model {slot + 1} · {caption}") for im, caption in fitted.overlays]
                        overlays += len(fitted.overlays)
                try:
                    others = [o.mesh for o in candidates if o.ok and o.index != c.index]
                    confidence[slot] = surface_confidence(mesh, c.mesh, others, self.reconstructor.camera, pinned)
                except Exception as exc:  # noqa: BLE001 - the heatmap is an extra: never lose the model over it
                    LOG.exception("Confidence map failed")
                    warnings.append(f"Model {slot + 1}: no confidence heatmap ({type(exc).__name__}: {exc}).")
                if opts.front_back_symmetric:
                    yield StageEvent("interior", "progress", f"Model {slot + 1}: rebuilding the back as a mirror of the front", fraction=slot / len(finals))
                    mirrored, reason = mirror_front_back(mesh)
                    if mirrored is None:
                        warnings.append(f"Model {slot + 1}: front-back symmetry skipped ({reason}).")
                    else:
                        mesh, remap, symmetry[slot] = mirrored.mesh, mirrored.remap, mirrored.stats
                image = variations[c.variation].image
                zero_shot = None
                if opts.interior == "auto" and hasattr(self.clip, "zero_shot"):
                    self._use("clip")
                    zero_shot = lambda labels, image=image: self.clip.zero_shot(image, labels)  # noqa: E731
                decision = decide_interior(opts.prompt, opts.interior, zero_shot)
                if decision.hollow:
                    yield StageEvent("interior", "progress", f"Model {slot + 1}: carving the cavity ({decision.describe()})", fraction=slot / len(finals))
                    hollowed, reason = hollow_out(mesh, opts.wall_thickness, self.cfg.hollow.voxel_resolution, min(opts.smooth_iterations, 3))
                    if hollowed is None:
                        warnings.append(f"Model {slot + 1} was kept solid: {reason}.")
                    else:
                        outer = remap
                        mesh, stats = hollowed.mesh, hollowed.stats
                        remap = hollowed.remap if outer is None else (lambda pts, h=hollowed.remap, o=outer: o(h(pts)))
                shaped.append((c, mesh, decision, stats, remap))
        if overlays:
            yield StageEvent("interior", "progress", payload={"candidates": candidate_gallery})
        yield StageEvent("interior", "done", " · ".join(
            f"Model {i + 1}: {('fitted to guides, ' if guided[i].views else 'holes opened, ') if i in guided else ''}{'mirrored, ' if i in symmetry else ''}{'hollow' if s else 'solid'}" for i, (_, _, _, s, _) in enumerate(shaped)
        ))

        # 6. Refine: decimate + optional UV texture bake ----------------------------------
        yield StageEvent("refine", "start", "Baking UV textures" if opts.bake_texture else "Finalising meshes (vertex colours)")
        assets = []
        with timer("refine"):
            for slot, (c, mesh, decision, stats, remap) in enumerate(shaped):
                yield StageEvent("refine", "progress", f"Model {slot + 1}/{len(shaped)}", fraction=slot / len(shaped))
                asset = MeshAsset.from_trimesh(decimate(mesh, self.cfg.mesh.max_faces))
                if opts.bake_texture:
                    asset = self._bake(asset, c.handle, opts.texture_resolution, warnings, remap)
                asset_path = build_dir / f"model{slot + 1}_{variations[c.variation].level.name}.npz"
                asset.save(asset_path)
                if slot in confidence:  # stored by position, so it survives a later bake that re-indexes vertices
                    points = asset.vertices.astype(np.float64)
                    save_confidence(_confidence_path(asset_path), points, confidence[slot].at(remap(points) if remap else points))
                self._remember(self._handles, str(asset_path), (c.handle, remap), 8)
                assets.append((asset_path, asset))
        for c in candidates:  # release candidate meshes/handles we no longer need
            c.mesh = c.handle = None
        yield StageEvent("refine", "done", " · ".join(f"{len(a.faces):,} faces" for _, a in assets) + " · " + (f"{opts.texture_resolution}px UV textures" if opts.bake_texture else "vertex colours"))

        # 7. Export + validation --------------------------------------------------------
        yield StageEvent("export", "start", "Writing GLB/OBJ for every model and re-opening them to validate")
        outputs: List[FinalOutput] = []
        with timer("export"):
            for slot, ((asset_path, asset), (c, _, decision, stats, _)) in enumerate(zip(assets, shaped)):
                bundle = _export(asset, asset_path, opts.depth_scale)
                section = section_image(asset.vertices, asset.faces)
                section.save(build_dir / f"{asset_path.stem}_section.png")
                conf = confidence.get(slot)
                outputs.append(FinalOutput(slot, c.variation, variations[c.variation].level.name, c.index, asset_path, bundle, decision, stats, section, c.score,
                                           symmetry.get(slot), guided.get(slot), conf.stats if conf else None))
                if not bundle.report["passed"]:
                    warnings.append(f"Model {slot + 1}: some export validation checks failed - see its validation report.")
            all_zip = zip_bundles([o.bundle for o in outputs], build_dir / f"unveil_{img.run_id}_all_depth{opts.depth_scale:.2f}.zip")

        gen_cfg = self.cfg.generation
        timings = dict(img.timings, **timer.timings)
        summary = {
            "run_id": img.run_id,
            "device": self.device,
            "low_vram": self.low_vram,
            "options": asdict(opts),
            "prompt_used": img.prompt,
            "models": {
                "base": gen_cfg.base_model,
                "controlnet": gen_cfg.controlnets[img.opts.control_mode],
                "clip": self.cfg.scoring.clip_model,
                "rembg": self.cfg.rembg_model,
                "reconstruction": f"{self.reconstructor.name} ({self.cfg.reconstruction.model_id})",
            },
            "variations": img.score_rows(),
            "variation_ranking": img.ranking,
            "selection": {"variations": selection, "selected_by": selected_by},
            "guide_views": {"given": guides.given, "side_front": guides.side_front},
            "candidates": [c.summary() for c in candidates],
            "outputs": [o.summary() for o in outputs],
            "timings_s": dict(timings, total=round(sum(timings.values()), 3)),
            "validation": [o.bundle.report for o in outputs],
            "warnings": warnings,
            "versions": _versions(),
        }
        write_json(summary, build_dir / "run.json")
        write_json(summary, img.run_dir / "run.json")
        LOG.info("Run %s build %s finished in %.1fs -> %s", img.run_id, build_dir.name, summary["timings_s"]["total"], [str(o.bundle.glb) for o in outputs])
        passed = all(o.bundle.report["passed"] for o in outputs)
        yield StageEvent("export", "done", f"{len(outputs)} model(s) · " + ("validation passed" if passed else "validation found issues"))
        yield RunResult(img.run_id, img.run_dir, build_dir, outputs, all_zip, selection, img.prep_images, img.variation_gallery, candidate_gallery, summary, warnings)

    def _bake(self, asset: MeshAsset, handle: Any, resolution: int, warnings: List[str], remap: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> MeshAsset:
        if not self.reconstructor.supports_color_query or handle is None:
            warnings.append("Texture baking is not supported by this backend; using vertex colours.")
            return asset
        remap = remap or (lambda pts: pts)
        try:
            self._use("reconstructor")
            return bake_texture(asset, lambda pts: self.reconstructor.query_color(handle, remap(pts)), resolution)
        except Exception as exc:  # noqa: BLE001 - vertex colours are a safe fallback
            LOG.exception("Texture bake failed")
            warnings.append(f"Texture bake failed ({type(exc).__name__}: {exc}); using vertex colours.")
            return asset

    # ------------------------------------------------------------------ CAD drawing -> 3D
    def _cad(self, path: Path, depth_scale: float) -> Iterator[Union[StageEvent, CadRun]]:
        run_id = new_run_id()
        build_dir = Path(self.cfg.output_dir) / run_id / "cad"
        build_dir.mkdir(parents=True, exist_ok=True)
        timer = StageTimer()
        is_dwg = path.suffix.lower() == ".dwg"

        yield StageEvent("cad_read", "start", "Converting DWG → DXF and reading it" if is_dwg else "Reading the DXF")
        with timer("cad_read"):
            doc = cad.read_drawing(path)
            curves = cad.extract_curves(doc)
        hidden = sum(c.hidden for c in curves)
        yield StageEvent("cad_read", "done", f"{len(curves)} curves ({hidden} hidden)")

        yield StageEvent("cad_views", "start", "Finding the front, top and side views")
        with timer("cad_views"):
            views, projection, source, warnings = cad.find_views(curves)
        yield StageEvent("cad_views", "done", f"{projection} projection, views from {'layer names' if source == 'layers' else 'their layout'}")

        yield StageEvent("cad_solid", "start", "Intersecting the three outlines and cutting holes")
        with timer("cad_solid"):
            mesh, holes, sil, dims = cad.build_solid(views, self.cfg.cad.resolution)
            model = cad.assemble(doc, curves, views, projection, source, mesh, holes, sil, dims, warnings)
        w, d, h = model.dimensions
        yield StageEvent("cad_solid", "done", f"{w:.4g} × {d:.4g} × {h:.4g} {model.units} · {len(holes)} hole(s)",
                         payload={"cad_images": [(model.preview, "Detected views (hidden lines dashed)")] + [(img, f"{name} silhouette" + (" · holes in red" if any(x.view == name for x in holes) else "")) for name, img in model.silhouettes.items()]})

        yield StageEvent("export", "start", "Writing GLB/OBJ and re-opening them to validate")
        with timer("export"):
            asset = MeshAsset.from_trimesh(decimate(model.mesh, self.cfg.cad.max_faces))  # flat faces: few triangles needed
            asset_path = build_dir / "model1_cad.npz"
            asset.save(asset_path)
            bundle = export_asset(asset, build_dir, depth_scale, asset_path.stem)
            section = section_image(asset.vertices, asset.faces)
            all_zip = zip_bundles([bundle], build_dir / f"unveil_{run_id}_all_depth{depth_scale:.2f}.zip")
        summary = {
            "run_id": run_id,
            "source": path.name,
            "cad": model.summary(),
            "timings_s": dict(timer.timings, total=round(sum(timer.timings.values()), 3)),
            "validation": [bundle.report],
            "versions": _versions(),
        }
        write_json(summary, build_dir / "run.json")
        if not bundle.report["passed"]:
            model.warnings.append("Some export validation checks failed - see the validation report.")
        yield StageEvent("export", "done", f"{len(asset.faces):,} faces · " + ("validation passed" if bundle.report["passed"] else "validation found issues"))
        yield CadRun(run_id, build_dir, path.name, model, asset_path, bundle, all_zip, section, summary, model.warnings)

    # ------------------------------------------------------------------ post-run actions
    @staticmethod
    def _build_assets(build_dir: Path) -> List[Path]:
        paths = sorted(Path(build_dir).glob("model*.npz"))
        if not paths:
            raise PipelineError("export", "No mesh found for this run - generate a model first.")
        return paths

    def reexport(self, build_dir: Path, depth_scale: float) -> Tuple[List[ExportBundle], Path]:
        """Re-export every model of a build with a new depth scale (geometry change, no re-inference)."""
        build_dir = Path(build_dir)
        bundles = [_export(MeshAsset.load(p), p, depth_scale) for p in self._build_assets(build_dir)]
        return bundles, zip_bundles(bundles, build_dir / f"unveil_{build_dir.parent.name}_all_depth{depth_scale:.2f}.zip")

    def bake_existing(self, build_dir: Path, depth_scale: float, resolution: int = 1024) -> Tuple[List[ExportBundle], Path, List[str]]:
        warnings: List[str] = []
        for asset_path in self._build_assets(build_dir):
            asset = MeshAsset.load(asset_path)
            if asset.textured:
                continue
            handle, remap = self._handles.get(str(asset_path), (None, None))
            if handle is None:
                raise PipelineError("refine", "This run's colour field is no longer in memory; re-run with 'Bake UV texture' enabled.")
            asset = self._bake(asset, handle, resolution, warnings, remap)
            asset.save(asset_path)
        bundles, all_zip = self.reexport(build_dir, depth_scale)
        return bundles, all_zip, warnings


def _confidence_path(asset_path: Path) -> Path:
    return asset_path.parent / f"confidence_{asset_path.stem}.npz"  # not "model*.npz": that pattern lists the models


def _export(asset: MeshAsset, asset_path: Path, depth_scale: float) -> ExportBundle:
    """GLB/OBJ export plus the confidence heatmap OBJ when the model has a confidence map."""
    bundle = export_asset(asset, asset_path.parent, depth_scale, asset_path.stem)
    bundle.heatmap = export_heatmap(asset, _confidence_path(asset_path), asset_path.parent / f"{bundle.glb.stem}_confidence.obj", depth_scale)
    return bundle


def _guide_masks(guides: GuideViews) -> Dict[str, np.ndarray]:
    """Filled silhouettes of the guide sketches."""
    masks = {}
    for name in guides.given:
        try:
            masks[name] = preprocess_sketch(getattr(guides, name), size=256).silhouette
        except SketchError as exc:
            raise PipelineError("interior", f"The {name}-view sketch could not be read: {exc}") from exc
    return masks


def _side_by_side(left: Optional[Image.Image], right: Image.Image) -> Image.Image:
    """TripoSR input framing next to its verification overlay."""
    h = right.height
    if left is None:
        return right
    left = left.convert("RGB").resize((h, h))
    out = Image.new("RGB", (left.width + right.width + 8, h), (255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(right.convert("RGB"), (left.width + 8, 0))
    return out
