"""End-to-end orchestration.

`UnveilPipeline.run()` is a generator: it yields `StageEvent`s while it works (the UI turns them into a
live progress board) and finally yields a `RunResult`. Every model is lazily loaded on first use.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import trimesh
from PIL import Image

from .background import BackgroundRemover
from .config import UnveilConfig
from .generation import SketchImageGenerator, Variation, build_prompt
from .mesh import (
    CandidateScore,
    CleanStats,
    ExportBundle,
    MeshAsset,
    MeshError,
    bake_texture,
    clean_mesh,
    decimate,
    export_asset,
    score_candidate,
    smooth_mesh,
)
from .preprocess import SketchError, preprocess_sketch
from .reconstruction import ReconstructionError, create_reconstructor
from .scoring import ClipScorer, rank_variations, score_variations
from .utils import LOG, StageTimer, flatten_alpha, free_gpu_memory, gpu_total_gb, new_run_id, resolve_device, write_json

STAGES: List[Tuple[str, str]] = [
    ("preprocess", "Sketch preprocessing"),
    ("generate", "ControlNet variations"),
    ("select_image", "Variation scoring & selection"),
    ("reconstruct", "TripoSR candidates & verification"),
    ("refine", "Mesh refinement & texture"),
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
class RunResult:
    run_id: str
    run_dir: Path
    bundle: ExportBundle
    prep_images: List[Tuple[Image.Image, str]]
    variation_gallery: List[Tuple[Image.Image, str]]
    candidate_gallery: List[Tuple[Image.Image, str]]
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
        self._handles: "OrderedDict[str, Any]" = OrderedDict()

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

    def _remember_handle(self, run_id: str, handle: Any) -> None:
        self._handles[run_id] = handle
        while len(self._handles) > 4:
            self._handles.popitem(last=False)

    # ------------------------------------------------------------------ main entry point
    def run(self, sketch: Image.Image, opts: RunOptions) -> Iterator[Union[StageEvent, RunResult]]:
        stage = "preprocess"
        try:
            for event in self._run(sketch, opts):
                if isinstance(event, StageEvent):
                    stage = event.stage
                yield event
        except (SketchError, PipelineError):
            raise
        except Exception as exc:  # noqa: BLE001 - convert to a stage-tagged, user-readable error
            if _is_oom(exc):
                free_gpu_memory()
                raise PipelineError(stage, "GPU ran out of memory. Try fewer variations, a lower marching-cubes resolution, or launch with --low-vram.") from exc
            LOG.exception("Stage %s failed", stage)
            raise PipelineError(stage, f"{type(exc).__name__}: {exc}") from exc

    def _run(self, sketch: Image.Image, opts: RunOptions) -> Iterator[Union[StageEvent, RunResult]]:
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
        yield StageEvent("generate", "done", f"{len(variations)} images generated", payload={"variations": [(v.image, f"{v.level.name} · seed {v.seed}") for v in variations]})

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
        variation_gallery = []
        for v, s in zip(variations, scores):
            mark = "★ " if v.index == top[0] else ("☆ " if v.index in top else "")
            variation_gallery.append((v.image, f"{mark}{v.level.name} · total {s.total:.2f} · IoU {s.iou:.2f} · CLIP {s.clip:.2f} · clean {s.clean:.2f}"))
        yield StageEvent("select_image", "done", f"Best: {variations[top[0]].level.name}", payload={"variations": variation_gallery})

        # 4. TripoSR candidates -> cleanup -> verification -------------------------------
        jobs = [(vi, ratio) for vi in top for ratio in opts.foreground_ratios]
        yield StageEvent("reconstruct", "start", f"Reconstructing {len(jobs)} candidates with TripoSR")
        candidates: List[Candidate] = []
        with timer("reconstruct"):
            self._use("reconstructor")
            for ci, (vi, ratio) in enumerate(jobs):
                yield StageEvent("reconstruct", "progress", f"Candidate {ci + 1}/{len(jobs)} · variation {variations[vi].level.name}, foreground {ratio:.2f}", fraction=ci / len(jobs))
                cand = Candidate(ci, vi, ratio)
                try:
                    cand.input_image = self.reconstructor.prepare_input(rgbas[vi], ratio)
                    result = self.reconstructor.reconstruct(cand.input_image, opts.mc_resolution, self.cfg.reconstruction.mc_threshold)
                    mesh, cand.stats = clean_mesh(result.mesh)
                    cand.mesh = smooth_mesh(mesh, opts.smooth_iterations)
                    cand.handle = result.handle
                    cand.score, cand.overlay = score_candidate(cand.mesh, prep.silhouette, self.reconstructor.camera, cand.stats, self.cfg.verify)
                except (ReconstructionError, MeshError, ValueError) as exc:
                    cand.error = str(exc)
                    warnings.append(f"Candidate {ci + 1} failed: {exc}")
                candidates.append(cand)
        best = select_best(candidates)
        if best is None:
            raise PipelineError("reconstruct", "All TripoSR candidates failed. Try a sketch with a clearer closed outline.")
        if not best.score.valid:
            warnings.append(f"No candidate passed validity checks ({best.score.reason}); returning the highest-scoring one.")
        candidate_gallery = []
        for c in candidates:
            if c.ok:
                mark = "★ " if c is best else ""
                s = c.score
                caption = f"{mark}#{c.index + 1} {variations[c.variation].level.name} @ {c.foreground_ratio:.2f} · total {s.total:.2f} · IoU {s.iou:.2f} · edge {s.edge:.2f} · kept {s.retention:.0%}"
                if not s.valid:
                    caption += f" · invalid: {s.reason}"
                candidate_gallery.append((c.overlay, caption))
                c.overlay.save(run_dir / f"candidate_{c.index}_overlay.png")
            else:
                candidate_gallery.append((Image.new("RGB", (256, 256), (245, 245, 245)), f"#{c.index + 1} failed: {c.error}"))
        yield StageEvent("reconstruct", "done", f"Best: candidate #{best.index + 1}", payload={"candidates": candidate_gallery})

        # 5. Refine: decimate + optional UV texture bake ----------------------------------
        yield StageEvent("refine", "start", "Baking UV texture" if opts.bake_texture else "Finalising mesh (vertex colours)")
        with timer("refine"):
            asset = MeshAsset.from_trimesh(decimate(best.mesh, self.cfg.mesh.max_faces))
            if opts.bake_texture:
                asset = self._bake(asset, best.handle, opts.texture_resolution, warnings)
            asset.save(run_dir / "asset.npz")
        self._remember_handle(run_id, best.handle)
        for c in candidates:  # release candidate meshes/handles we no longer need
            c.mesh = c.handle = None
        yield StageEvent("refine", "done", f"{len(asset.faces):,} faces · " + (f"{opts.texture_resolution}px UV texture" if asset.textured else "vertex colours"))

        # 6. Export + validation --------------------------------------------------------
        yield StageEvent("export", "start", "Writing GLB/OBJ and re-opening them to validate")
        with timer("export"):
            bundle = export_asset(asset, run_dir, opts.depth_scale)
        if not bundle.report["passed"]:
            warnings.append("Some export validation checks failed - see the validation report.")

        summary = {
            "run_id": run_id,
            "device": self.device,
            "low_vram": self.low_vram,
            "options": asdict(opts),
            "prompt_used": prompt,
            "models": {
                "base": gen_cfg.base_model,
                "controlnet": gen_cfg.controlnets[opts.control_mode],
                "clip": self.cfg.scoring.clip_model,
                "rembg": self.cfg.rembg_model,
                "reconstruction": f"{self.reconstructor.name} ({self.cfg.reconstruction.model_id})",
            },
            "variations": [dict(asdict(s), level=variations[s.index].level.name, seed=variations[s.index].seed) for s in scores],
            "variation_ranking": ranking,
            "candidates": [c.summary() for c in candidates],
            "selected_candidate": best.index,
            "mesh": {"vertices": int(len(asset.vertices)), "faces": int(len(asset.faces)), "textured": asset.textured},
            "timings_s": dict(timer.timings, total=timer.total),
            "validation": bundle.report,
            "warnings": warnings,
            "versions": _versions(),
        }
        write_json(summary, run_dir / "run.json")
        LOG.info("Run %s finished in %.1fs -> %s", run_id, timer.total, bundle.glb)
        yield StageEvent("export", "done", "Validation passed" if bundle.report["passed"] else "Validation found issues")
        yield RunResult(run_id, run_dir, bundle, prep_images, variation_gallery, candidate_gallery, summary, warnings)

    def _bake(self, asset: MeshAsset, handle: Any, resolution: int, warnings: List[str]) -> MeshAsset:
        if not self.reconstructor.supports_color_query or handle is None:
            warnings.append("Texture baking is not supported by this backend; using vertex colours.")
            return asset
        try:
            self._use("reconstructor")
            return bake_texture(asset, lambda pts: self.reconstructor.query_color(handle, pts), resolution)
        except Exception as exc:  # noqa: BLE001 - vertex colours are a safe fallback
            LOG.exception("Texture bake failed")
            warnings.append(f"Texture bake failed ({type(exc).__name__}: {exc}); using vertex colours.")
            return asset

    # ------------------------------------------------------------------ post-run actions
    def reexport(self, run_dir: Path, depth_scale: float) -> ExportBundle:
        """Re-export a finished run with a new depth scale (geometry change, no re-inference)."""
        asset_path = Path(run_dir) / "asset.npz"
        if not asset_path.exists():
            raise PipelineError("export", "No mesh found for this run - generate a model first.")
        return export_asset(MeshAsset.load(asset_path), Path(run_dir), depth_scale)

    def bake_existing(self, run_id: str, run_dir: Path, depth_scale: float, resolution: int = 1024) -> Tuple[ExportBundle, List[str]]:
        asset_path = Path(run_dir) / "asset.npz"
        asset = MeshAsset.load(asset_path)
        warnings: List[str] = []
        if not asset.textured:
            handle = self._handles.get(run_id)
            if handle is None:
                raise PipelineError("refine", "This run's colour field is no longer in memory; re-run with 'Bake UV texture' enabled.")
            asset = self._bake(asset, handle, resolution, warnings)
            asset.save(asset_path)
        return export_asset(asset, Path(run_dir), depth_scale), warnings
