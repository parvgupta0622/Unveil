# Unveil — from a 2D sketch to a 3D asset

Unveil turns a rough single-view sketch (drawn in the app or uploaded) and an optional text prompt into a
coloured, exportable 3D model (**GLB** and **OBJ+MTL+PNG**). It runs on a **free Google Colab T4**.

```
sketch ─► preprocessing ─► SD1.5 + ControlNet ×4 creativity levels ─► score & pick (IoU + CLIP + clean-bg)
       ─► background removal ─► TripoSR ×4 candidates ─► largest component + smoothing
       ─► reproject & verify vs. sketch ─► best mesh ─► UV texture bake ─► depth (Z) scaling ─► GLB/OBJ ─► re-open & validate
```

Built for the TechZephyr 4.0 *Sketch2Shape AI* problem statement.

---

## Contents
1. [Quick start (Colab)](#1-quick-start-google-colab)
2. [Using the app](#2-using-the-app)
3. [Architecture](#3-architecture)
4. [Key features in detail](#4-key-features-in-detail)
5. [Models, weights and licences](#5-models-weights-and-licences)
6. [Hardware, runtime and memory](#6-hardware-runtime-and-memory)
7. [Local installation & CLI](#7-local-installation--cli)
8. [Outputs](#8-outputs)
9. [Configuration](#9-configuration)
10. [Testing](#10-testing)
11. [Extending: adding another reconstruction model](#11-extending-adding-another-reconstruction-model)
12. [Limitations](#12-limitations)
13. [Troubleshooting](#13-troubleshooting)
14. [Project structure](#14-project-structure)

---

## 1. Quick start (Google Colab)

1. Open `notebooks/Unveil_Colab.ipynb` in Colab (File → Upload notebook, or open it from GitHub).
2. **Runtime → Change runtime type → T4 GPU.**
3. Run the cells in order:
   | Cell | What it does | Time |
   |---|---|---|
   | 1 | Shows the GPU | – |
   | 2 | Clones `https://github.com/parvgupta0622/Unveil` | seconds |
   | 3 | `bash scripts/setup_colab.sh`: pip installs, clones TripoSR at a pinned commit, checks imports | ~2 min |
   | 4 | *(optional)* pre-downloads all weights (~6 GB) | 3–6 min |
   | 5 | `python app.py --share` → open the printed `*.gradio.live` link | – |

No Hugging Face token is needed: every model is public and ungated.

## 2. Using the app

1. **Input.** Either *Draw* (brush, eraser, undo, **Reset canvas**) or *Upload* a PNG/JPG. Photos of paper sketches work too: uneven lighting is flattened and noise specks are removed. Four example sketches are in the *Upload* tab.
2. **Prompt (optional).** Describe the object, its material, colour or style. Without a prompt the pipeline still works; image selection then uses CLIP similarity to the sketch itself.
3. **Sketch interpretation.**
   - **Precise**: lineart ControlNet; follows your lines closely.
   - **Freeform**: scribble ControlNet; tolerant of rough, wobbly or open strokes.
4. **Bake UV texture** (default on): bakes a UV texture so colour survives in OBJ and in every DCC tool. Turn it off for faster vertex-colour output; you can still bake later with the *Bake UV texture for this model* button.
5. Press **Generate 3D**. A stage board shows each step live, with timings.
6. **Inspect.** The 3D viewer supports drag-to-orbit and scroll-to-zoom. The tabs show the preprocessing steps, all ControlNet variations with their scores, and every 3D candidate's reprojection overlay (blue = mesh, red = sketch outline, ★ = chosen). *Run details* shows every score, timing and package version.
7. **Depth.** Choose *Thin / Medium / Thick* or drag the slider (0.3×–2.5×). This **rescales the real geometry** along the view axis and **re-exports** the GLB/OBJ, with the validation report refreshed. No re-inference is needed.
8. **Download** the GLB, or the OBJ zip (`.obj` + `.mtl` + texture `.png`).

The *Advanced settings* panel exposes the seed, number of variations (3–4), diffusion steps, CFG, how many images go to TripoSR (1–2), marching-cubes resolution, smoothing, texture resolution and an extra negative prompt.

## 3. Architecture

| Stage | Module | What happens |
|---|---|---|
| Input & preprocessing | `unveil/preprocess.py` | Flatten alpha → auto-invert dark backgrounds → Otsu threshold, or illumination-flattened Otsu for photos → speck removal → crop and centre in a 512² square → skeletonise + spur pruning → per-mode stroke width. A filled **silhouette** is built by progressive gap closing + hole filling, falling back to a convex hull. Solid (filled) drawings are detected and handled separately. |
| Sketch-guided generation | `unveil/generation.py` | DreamShaper-8 (SD1.5) + ControlNet v1.1 *lineart* or *scribble*, UniPC, 20 steps, fp16. One pipeline, with ControlNets hot-swapped per mode. |
| Creativity ladder & selection | `unveil/scoring.py` | 3–4 variations (see §4.2), each scored by sketch-silhouette IoU, CLIP and a clean-background check. |
| Foreground isolation | `unveil/background.py` | rembg (U²-Net), alpha-island cleanup, and a border-colour-keying fallback if rembg is unavailable. TripoSR framing (foreground ratio, 50 % grey). |
| Image → 3D | `unveil/reconstruction/` | `Reconstructor` interface + registry. **TripoSR** is the only backend. Meshes are returned in a shared *view frame* (+x right, +y up, +z towards the camera). |
| Mesh processing | `unveil/mesh/processing.py` | Weld, drop degenerate faces, **keep the largest connected component**, fix winding (signed volume), Taubin smoothing, quadric decimation (> 120 k faces). |
| Verification | `unveil/mesh/verify.py` | Perspective reprojection into TripoSR's input camera, silhouette IoU + edge chamfer vs. the sketch, floater penalty, validity checks. |
| Texture | `unveil/mesh/texture.py` | xatlas UV unwrap → CPU UV rasterisation → barycentric lift to 3D → query TripoSR's colour field → nearest-texel seam padding. No OpenGL. |
| Export & validation | `unveil/mesh/export.py` | Depth scaling, centring, GLB (PBR material) + OBJ/MTL/PNG + zip. Each file is **re-opened** with trimesh to check that it opens, has non-empty finite geometry, keeps its face count, has 3D extent, and keeps its texture or vertex colours. |
| Orchestration | `unveil/pipeline.py` | Generator of `StageEvent`s → `RunResult`. Lazy model loading, GPU residency management, stage-tagged errors, OOM handling, `run.json`. |
| UI / CLI | `unveil/ui.py`, `app.py`, `unveil/cli.py` | Gradio 5 Blocks UI; headless CLI with identical behaviour. |

## 4. Key features in detail

### 4.1 Multi-candidate 3D generation and verification
- **Where candidates come from.** TripoSR is deterministic, so diversity comes from its inputs: the **top-2 ControlNet images × 2 foreground framings (0.75, 0.85)** give **4 TripoSR runs**.
- **Cleanup.** Each mesh keeps only its **largest connected component** (removes floaters). The fraction of faces kept (*retention*) is recorded as a quality signal. Winding is fixed and Taubin smoothing applied.
- **Verification.** Each mesh is perspective-projected with TripoSR's canonical input camera (distance 1.9, FOV 40°). Its silhouette is compared with the sketch silhouette after both are normalised to their bounding boxes:
  - `score = 0.5·IoU + 0.3·exp(−chamfer/σ) + 0.2·retention`, where *chamfer* is the symmetric mean distance between the two outlines and σ = 3 % of the canvas.
- **Validity.** A candidate is invalid if it has fewer than 500 faces, is not visible from the input view, or is *flat* (depth < 4 % of width/height). The highest-scoring valid candidate wins. If none are valid, the best one is returned with a warning, and failed candidates are reported rather than crashing the run.

### 4.2 Creativity-controlled generation
| Level | ControlNet conditioning scale | Guidance end | Seed |
|---|---|---|---|
| strict | 1.3 | 1.00 | seed |
| balanced | 1.0 | 0.90 | seed+1 |
| creative | 0.8 | 0.80 | seed+2 |
| free | 0.6 | 0.65 | seed+3 |

Each image is scored with
`0.45·IoU(rembg mask, sketch silhouette) + 0.35·CLIP + 0.20·clean`.
- **CLIP** is ViT-B/32 cosine similarity to `"a photo of <prompt>"`, rescaled to 0–1. With no prompt, it is image-to-image similarity with the cleaned sketch.
- **clean** is the mean of four checks: object area in a sensible range, single connected object, object not touching the border, and uniform background.
- Variations with IoU < 0.35 are ranked below every variation that preserves the sketch structure.

### 4.3 Z-axis depth control
Single-view reconstruction often misjudges thickness. The depth control scales vertices along the view axis about the mesh's own centre. It changes the **actual exported geometry** (the GLB/OBJ files are rewritten, and the validation report shows the new W×H×D). It is not a viewer effect. Presets: Thin 0.5×, Medium 1.0×, Thick 1.6×; slider 0.3–2.5×. Re-export takes about a second because the cleaned mesh and texture are cached in `outputs/<run>/asset.npz`.

## 5. Models, weights and licences

All weights download automatically from the Hugging Face Hub on first use (cached in `~/.cache/huggingface`). rembg's U²-Net goes to `~/.u2net`.

| Component | Model | Size (approx.) | Licence |
|---|---|---|---|
| Base diffusion model | [`Lykon/dreamshaper-8`](https://huggingface.co/Lykon/dreamshaper-8) (SD 1.5 fine-tune), fp16 | 2.0 GB | CreativeML OpenRAIL-M |
| ControlNet "Precise" | [`lllyasviel/control_v11p_sd15_lineart`](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart) | 1.4 GB | OpenRAIL |
| ControlNet "Freeform" | [`lllyasviel/control_v11p_sd15_scribble`](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble) | 1.4 GB | OpenRAIL |
| Image scoring | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | 0.6 GB | MIT |
| Background removal | U²-Net via [rembg](https://github.com/danielgatis/rembg) | 0.17 GB | Apache-2.0 / MIT |
| Image → 3D | [`stabilityai/TripoSR`](https://huggingface.co/stabilityai/TripoSR) + [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) code @ `107cefd` (uses `facebook/dino-vitb16` config) | 1.7 GB | MIT |

Third-party libraries: PyTorch, Diffusers, Transformers, Accelerate, trimesh, xatlas, fast-simplification, scikit-image, SciPy, OpenCV, Pillow, rembg/onnxruntime, Gradio. No external APIs or paid services are used.

**Why TripoSR?** InstantMesh and Unique3D exceed the hardware available to us, so this version uses TripoSR as its only reconstruction model. It is a feed-forward model that takes about a second per image and fits a T4 alongside SD1.5. The reconstruction layer is a pluggable interface (§11).

**torchmcubes.** TripoSR's marching cubes normally needs `torchmcubes`, a CUDA extension compiled from source. Unveil installs a drop-in **scikit-image shim** when it is absent, so no compile step is needed. A unit test checks that the shim returns surfaces in exactly the axes TripoSR samples density in. If you do install `torchmcubes`, it is picked up automatically.

## 6. Hardware, runtime and memory

**Target:** Google Colab free tier, NVIDIA **T4 (15 GB VRAM)**, ~12 GB system RAM, ~10 GB disk for weights.

**Expected time per model on a T4, default settings (4 variations, 4 candidates, 256³ marching cubes, 1024² texture):**

| Stage | Expected |
|---|---|
| Preprocessing | < 1 s |
| 4 × ControlNet images (512², 20 steps) | ~15–25 s |
| Background removal + CLIP scoring | ~3–6 s |
| 4 × TripoSR + cleanup + verification | ~25–40 s |
| Decimation + UV bake | ~10–30 s |
| Export + validation | ~2–5 s |
| **Total (warm)** | **~1–2 min** |
| First run (weight download + load) | +3–6 min |

> ⚠️ These figures are **estimates** from published per-model speeds. The first version of this code was developed without GPU access (see §10). Every run records its exact per-stage timings in `outputs/<run>/run.json` and in the *Run details* tab. **Replace this table with measured numbers after the first Colab run.**

**VRAM.** Everything is fp16 except TripoSR (fp32). All models stay resident (~6 GB of weights), with an expected peak of ~9–11 GB during TripoSR mesh extraction. On GPUs with < 13 GB, idle models are automatically moved to the CPU between stages; force this with `--low-vram`.

**Ways to go faster:** 3 variations, *Images sent to TripoSR* = 1 (2 candidates), marching cubes 192, texture bake off.

## 7. Local installation & CLI

Requires Python 3.10+, git, and an NVIDIA GPU (8 GB+ with `--low-vram`). CPU/MPS also run, but slowly.

```bash
git clone https://github.com/parvgupta0622/Unveil.git && cd Unveil
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick the build for your CUDA
bash scripts/setup_colab.sh        # deps + TripoSR source (works outside Colab too)
python app.py                      # http://localhost:7860  (add --share for a public link)
```

Headless (same pipeline, same outputs):

```bash
python -m unveil.cli --sketch examples/mug.png --prompt "a glossy red ceramic mug" --mode precise --seed 42 --depth Medium
python -m unveil.cli --sketch my_sketch.jpg --no-texture --variations 3 --top-k 1
python -m unveil.cli --preload-only    # just download every model
```

`app.py` flags: `--share`, `--port`, `--host`, `--preload` (load all models at start-up), `--low-vram`, `--output-dir`.

## 8. Outputs

Each run writes `outputs/<YYYYMMDD-HHMMSS-xxxx>/`:

| File | Content |
|---|---|
| `unveil_depth1.00.glb` | Final model, PBR material with base-colour texture (or vertex colours) |
| `unveil_depth1.00_obj/` + `…_obj.zip` | `.obj` + `.mtl` + texture `.png` |
| `unveil_depth<X>.glb / _obj.zip` | Re-exports created by the depth control |
| `asset.npz` | Cleaned mesh (+UV/texture) used for depth re-exports |
| `prep_*.png`, `variation_*.png`, `candidate_*_overlay.png` | Intermediate results |
| `run.json` | Options, seeds, every score, the chosen variation/candidate, cleanup stats, timings, validation report, package versions |

Exported models are centred at the origin, Y-up, with the front facing +Z (glTF convention). Units are arbitrary, at about 1 unit tall.

## 9. Configuration

Every tunable lives in `unveil/config.py`: model IDs, the creativity ladder, prompt template, scoring and verification weights, IoU floor, foreground ratios, marching-cubes threshold, decimation limit, depth presets and so on. `UNVEIL_OUTPUT_DIR` and `UNVEIL_TRIPOSR_PATH` can be set as environment variables.

## 10. Testing

```bash
pip install pytest
python -m pytest -q tests
```

29 CPU-only tests (no model weights needed):
- **Preprocessing:** outline → filled silhouette, inverted inputs, photos with uneven lighting, filled shapes, open strokes, speck noise, recentring, empty canvas errors.
- **Mesh:** floater removal + winding repair, smoothing/decimation keeping colours, depth scaling changes only Z, projection orientation, silhouette scoring and flat-mesh rejection, texture bake correctness (colour sampled through the UVs), GLB/OBJ export + validation round-trips.
- **TripoSR adapter:** axis-convention rotation, and the marching-cubes shim run through TripoSR's own `MarchingCubeHelper` (needs the TripoSR source).
- **Pipeline flow:** the full orchestrator runs end to end with small deterministic stand-ins in place of the three neural networks. Everything else is real: preprocessing, background-removal fallback, scoring/ranking, cleanup, verification, bake, export, validation, depth re-export and post-hoc baking. The Gradio UI build is also tested.

**Verification status of this draft.** The test suite passes, and the Gradio UI was driven in a browser with the stand-in models: drawing, generation, progress board, galleries, 3D viewer orientation, depth presets and downloads. **The real SD/ControlNet/CLIP/TripoSR inference path has not yet been run on a GPU** (the development machine had none). The first Colab run is the integration test for those calls.

## 11. Extending: adding another reconstruction model

1. Create `unveil/reconstruction/<name>.py` with a subclass of `Reconstructor` (`unveil/reconstruction/base.py`) that implements `load`, `to`, `prepare_input`, `reconstruct` and a `camera` (`CameraSpec`). Optionally add `query_color` for texture baking.
2. Return meshes in the **view frame** (+x right, +y up, +z towards the input camera). Verification, depth control and export then work unchanged.
3. Register it in `BACKENDS` in `unveil/reconstruction/__init__.py` and set `ReconstructionConfig.backend`.

## 12. Limitations

- **Hidden geometry is guessed.** A single view cannot determine the back of the object. TripoSR's guess can be too thin or too thick, which is what the depth control is for.
- **TripoSR quality ceiling.** Its meshes are smooth but low in detail at 256³. Thin structures (chair legs, lamp stems) can come out thick, merged or broken, and the largest-component filter may then drop a detached part. Textures come from one view, so the back is extrapolated.
- **Sketch → silhouette is 2D.** Holes in the object (a mug handle's opening) are filled in the reference silhouette, and interior detail lines are ignored by the silhouette metrics.
- **Scoring weights are hand-tuned**, not learned. CLIP ViT-B/32 is a coarse semantic judge.
- **SD1.5 at 512²** limits image detail. SDXL was avoided to stay within T4 memory and time.
- **Not production geometry:** no guaranteed manifoldness, no retopology, arbitrary units. This is out of the problem's mandatory scope.
- `fast-simplification` decimation transfers vertex colours by nearest neighbour.

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Relaunch with `python app.py --share --low-vram`; use 3 variations, 1 image to TripoSR, marching cubes 192. |
| `No module named tsr` / TripoSR clone failed | Re-run `bash scripts/setup_colab.sh`, or `git clone https://github.com/VAST-AI-Research/TripoSR third_party/TripoSR`, or set `UNVEIL_TRIPOSR_PATH`. |
| `Missing key(s) in state_dict … image_tokenizer.model.layers…` | transformers 5 renamed ViT layers. Fixed in `unveil/reconstruction/triposr.py` (`adapt_state_dict` remaps the checkpoint). Pull the latest code. |
| pip prints "dependency resolver … conflicts" (google-genai, google-adk, fasthtml, hf-gradio) during setup | Harmless: these are Colab-preinstalled packages that Unveil does not use; the conflicts come from Gradio 5 pinning older `starlette`/`gradio-client`. Setup is fine if it ends with `Unveil setup OK`. |
| "The sketch looks empty" | Strokes too faint or tiny. Use a darker or thicker pen, or upload a higher-contrast scan. |
| Generated image ignores the sketch | Use **Precise** mode, and make sure the outline is closed. Check the *ControlNet variations* tab: IoU < 0.35 is flagged. |
| Mesh far too thin/thick | Use the depth presets or slider. |
| Chosen candidate looks wrong | See the *3D candidates* tab. Try a different seed, or set *Images sent to TripoSR* = 2. |
| rembg/onnxruntime errors | The pipeline falls back to border-colour keying automatically (a warning is logged). `pip install rembg onnxruntime` to restore. |
| Gradio link not printed / expired | Re-run the launch cell. Share links last 72 h. |
| Slow first run | One-time weight download (~6 GB). Run `scripts/download_weights.py` first. |

## 14. Project structure

```
Unveil/
├── app.py                         # Gradio launcher
├── requirements.txt
├── notebooks/Unveil_Colab.ipynb   # Colab entry point
├── scripts/setup_colab.sh         # deps + pinned TripoSR clone + import check
├── scripts/download_weights.py    # pre-fetch all weights
├── examples/                      # sample sketches
├── tests/                         # CPU test suite (pytest)
└── unveil/
    ├── config.py                  # all tunables
    ├── preprocess.py              # sketch cleanup, ControlNet conditioning, silhouette
    ├── generation.py              # SD1.5 + ControlNet, creativity ladder
    ├── scoring.py                 # IoU / CLIP / clean-background scoring + ranking
    ├── background.py              # rembg + fallback, framing
    ├── reconstruction/            # Reconstructor interface, registry, TripoSR backend
    ├── mesh/                      # asset, cleanup, verification, texture bake, export/validation
    ├── pipeline.py                # orchestration
    ├── ui.py                      # Gradio interface
    └── cli.py                     # headless runner
```
