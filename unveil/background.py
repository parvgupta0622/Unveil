"""Foreground isolation (rembg) with a deterministic fallback, plus foreground framing."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .utils import LOG


class BackgroundRemover:
    """Lazily-initialised rembg session. Falls back to border-colour keying if rembg is unavailable."""

    def __init__(self, model_name: str = "u2net") -> None:
        self.model_name = model_name
        self._session = None
        self._failed = False

    def load(self) -> None:
        if self._session is not None or self._failed:
            return
        try:
            import rembg

            self._session = rembg.new_session(self.model_name)
        except Exception as exc:  # noqa: BLE001 - any failure here should degrade, not crash
            self._failed = True
            LOG.warning("rembg unavailable (%s); using border-colour background keying", exc)

    def remove(self, image: Image.Image) -> Image.Image:
        """Return an RGBA image whose alpha isolates the main object."""
        rgb = image.convert("RGB")
        self.load()
        alpha = None
        if self._session is not None:
            try:
                import rembg

                alpha = np.asarray(rembg.remove(rgb, session=self._session).convert("RGBA"))[..., 3]
            except Exception as exc:  # noqa: BLE001
                LOG.warning("rembg failed on an image (%s); using fallback keying", exc)
        if alpha is None or (alpha > 127).sum() < 64:
            alpha = heuristic_alpha(np.asarray(rgb))
        alpha = clean_alpha(alpha)
        return Image.fromarray(np.dstack([np.asarray(rgb), alpha]))


def heuristic_alpha(rgb: np.ndarray, tol: float = 30.0) -> np.ndarray:
    """Key out the dominant border colour. Works well on the plain backgrounds we prompt for."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(np.float32)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=-1)
    mask = dist > tol
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    mask = ndi.binary_fill_holes(mask)
    return np.where(mask, 255, 0).astype(np.uint8)


def clean_alpha(alpha: np.ndarray, rel: float = 0.01) -> np.ndarray:
    """Drop small detached alpha islands; they would become floating geometry in 3D."""
    solid = (alpha > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n <= 2:
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= rel * areas.max()
    # Keep soft edges of retained components, zero everything belonging to dropped islands.
    dropped = ~keep[labels] & (labels > 0)
    dropped = cv2.dilate(dropped.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = alpha.copy()
    out[dropped] = 0
    return out


def resize_foreground(rgba: Image.Image, ratio: float) -> Image.Image:
    """Crop to the alpha bounding box, pad to a square, then pad so the object spans `ratio` of it."""
    arr = np.asarray(rgba.convert("RGBA"))
    ys, xs = np.nonzero(arr[..., 3] > 0)
    if ys.size == 0:
        raise ValueError("Foreground mask is empty")
    fg = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    side = max(fg.shape[:2])
    full = int(round(side / ratio))
    canvas = np.zeros((full, full, 4), np.uint8)
    oy, ox = (full - fg.shape[0]) // 2, (full - fg.shape[1]) // 2
    canvas[oy : oy + fg.shape[0], ox : ox + fg.shape[1]] = fg
    return Image.fromarray(canvas)


def composite_on(rgba: Image.Image, gray: float = 0.5) -> Image.Image:
    arr = np.asarray(rgba.convert("RGBA")).astype(np.float32) / 255.0
    rgb = arr[..., :3] * arr[..., 3:4] + (1.0 - arr[..., 3:4]) * gray
    return Image.fromarray((rgb * 255.0).round().astype(np.uint8))
