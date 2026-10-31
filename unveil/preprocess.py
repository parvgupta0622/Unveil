"""Sketch preprocessing.

Turns a canvas drawing or a photographed/scanned sketch into:
  * normalized centre-line strokes (ControlNet conditioning is rendered from these per mode),
  * a filled silhouette mask (used to score generated images and reconstructed meshes).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .utils import flatten_alpha, mask_to_image

# Stroke width (px at 512) of the ControlNet conditioning per mode. Lineart was trained on thin
# anti-aliased lines, scribble on thick marker-like strokes.
STROKE_WIDTH = {"precise": 3, "freeform": 7}

MIN_INK_PIXELS = 30
FILLED_RATIO = 0.35  # ink / bbox area above this => input is a solid shape, not line art


class SketchError(ValueError):
    """The input cannot be interpreted as a sketch (e.g. empty canvas)."""


@dataclass
class SketchPrep:
    strokes: np.ndarray  # bool (S, S) one-pixel centre lines
    ink: np.ndarray  # bool (S, S) binarized ink, original thickness
    silhouette: np.ndarray  # bool (S, S) object region implied by the sketch
    filled_input: bool
    size: int

    def control_image(self, mode: str = "precise") -> Image.Image:
        """White strokes on black, as expected by ControlNet v1.1 lineart / scribble."""
        lines = thicken(self.strokes, STROKE_WIDTH.get(mode, STROKE_WIDTH["precise"]))
        return mask_to_image(lines)

    def clean_image(self) -> Image.Image:
        """Black strokes on white, for display."""
        lines = thicken(self.strokes, 3)
        return Image.fromarray(np.where(lines, 0, 255).astype(np.uint8)).convert("RGB")

    def silhouette_image(self) -> Image.Image:
        return mask_to_image(self.silhouette)


def thicken(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def preprocess_sketch(image: Image.Image, size: int = 512, margin: float = 0.08) -> SketchPrep:
    if image is None:
        raise SketchError("No sketch provided - draw on the canvas or upload an image.")
    gray = np.asarray(flatten_alpha(image).convert("L"), dtype=np.uint8)
    if min(gray.shape) < 16:
        raise SketchError("Sketch image is too small.")
    gray = _rescale(gray, lo=512, hi=1024)
    if np.median(gray) < 110:  # light strokes on a dark background (chalkboard, inverted scan)
        gray = 255 - gray

    ink = _remove_specks(_binarize(gray))
    if ink.sum() < MIN_INK_PIXELS:
        raise SketchError("The sketch looks empty - draw or upload a sketch with visible strokes.")

    ink = _crop_to_square(ink, size, margin)
    if ink.sum() < MIN_INK_PIXELS:
        raise SketchError("The sketch strokes are too faint or too small to use.")

    ys, xs = np.nonzero(ink)
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    filled = ink.sum() / max(bbox_area, 1) > FILLED_RATIO

    if filled:
        silhouette = _keep_significant(ndi.binary_fill_holes(ink))
        edge = cv2.morphologyEx(silhouette.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        strokes = skeletonize(edge)
    else:
        strokes = _prune_spurs(skeletonize(ink), length=max(4, size // 80))
        silhouette = _fill_silhouette(ink)
    return SketchPrep(strokes=strokes, ink=ink, silhouette=silhouette, filled_input=bool(filled), size=size)


def _rescale(gray: np.ndarray, lo: int, hi: int) -> np.ndarray:
    side = max(gray.shape)
    if lo <= side <= hi:
        return gray
    scale = (lo if side < lo else hi) / side
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(gray, (round(gray.shape[1] * scale), round(gray.shape[0] * scale)), interpolation=interp)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Ink mask. Uses plain Otsu for evenly lit inputs, illumination-flattened Otsu for photos."""
    # No median filter here: it would erase one-pixel pen lines. Noise is handled by _remove_specks.
    if int(gray.max()) - int(gray.min()) < 40:
        return np.zeros(gray.shape, bool)
    otsu, raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    background = gray[raw == 0]
    if background.size and background.std() < 12:
        return gray < np.clip(otsu, 60, 230)

    # Uneven lighting (phone photo of paper): divide out a background estimate obtained by
    # closing away the dark strokes, then threshold the flattened image.
    k = max(15, (max(gray.shape) // 25) | 1)
    est = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.GaussianBlur(cv2.divide(gray, np.maximum(est, 1), scale=255), (3, 3), 0)
    otsu, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return norm < np.clip(otsu, 90, 215)


def _remove_specks(ink: np.ndarray) -> np.ndarray:
    min_area = max(8, int(0.00005 * ink.size))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def _crop_to_square(ink: np.ndarray, size: int, margin: float) -> np.ndarray:
    ys, xs = np.nonzero(ink)
    crop = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    side = int(np.ceil(max(crop.shape) / (1 - 2 * margin)))
    canvas = np.zeros((side, side), np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop.astype(np.uint8) * 255
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    # A low threshold after area-downsampling keeps one-pixel lines from vanishing.
    return cv2.resize(canvas, (size, size), interpolation=interp) > 50


_NEIGHBOURS = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.float32)


def _endpoints(skel: np.ndarray) -> np.ndarray:
    count = cv2.filter2D(skel.astype(np.float32), -1, _NEIGHBOURS, borderType=cv2.BORDER_CONSTANT)
    return np.isclose(count, 11)  # a skeleton pixel with exactly one neighbour


def _prune_spurs(skel: np.ndarray, length: int) -> np.ndarray:
    """Remove short skeleton branches caused by jagged/shaky stroke edges.

    Classic prune: erode endpoints `length` times, then regrow surviving line ends along the original
    skeleton. Small strokes that would disappear entirely (dots, short dashes) are restored.
    """
    pruned = skel.copy()
    for _ in range(length):
        ends = _endpoints(pruned)
        if not ends.any():
            break
        pruned &= ~ends
    grown = _endpoints(pruned)
    for _ in range(length):
        grown = (cv2.dilate(grown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & skel & ~pruned
    result = pruned | grown
    n, labels = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
    kept = np.zeros(n, bool)
    kept[np.unique(labels[result])] = True
    return result | (~kept[labels] & (labels > 0))


def _keep_significant(mask: np.ndarray, rel: float = 0.02) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= rel * areas.max()
    return keep[labels]


def _fill_silhouette(ink: np.ndarray) -> np.ndarray:
    """Fill the region enclosed by the strokes, closing progressively larger gaps if needed."""
    ink_u8 = ink.astype(np.uint8)
    for k in (3, 7, 11, 17, 25):
        closed = cv2.morphologyEx(ink_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        filled = ndi.binary_fill_holes(closed > 0)
        if filled.sum() > 2.0 * (closed > 0).sum():
            # Union with the ink so open parts (a handle drawn as a single arc) still count.
            return _keep_significant(filled | ink, rel=0.005)
    # Open line art that never encloses a region: fall back to the convex hull of all strokes.
    pts = np.column_stack(np.nonzero(ink)[::-1]).astype(np.int32)
    hull = cv2.convexHull(pts)
    out = np.zeros(ink.shape, np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out > 0
