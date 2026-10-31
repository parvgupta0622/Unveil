import numpy as np

from unveil.background import clean_alpha, heuristic_alpha, resize_foreground
from unveil.scoring import VariationScore, clean_background_score, rank_variations
from PIL import Image


def _object_image(clutter=False):
    rgb = np.full((256, 256, 3), 250, np.uint8)
    rgb[80:180, 90:170] = (200, 40, 40)
    if clutter:
        rgb[:] = np.random.default_rng(0).integers(0, 255, rgb.shape, dtype=np.uint8)
        rgb[80:180, 90:170] = (200, 40, 40)
    return rgb


def test_clean_background_prefers_plain_backgrounds():
    clean_rgb = _object_image()
    alpha = heuristic_alpha(clean_rgb)
    good, details = clean_background_score(clean_rgb, alpha)
    assert good > 0.9 and details["single_object"] == 1.0

    busy = _object_image(clutter=True)
    alpha_busy = np.zeros((256, 256), np.uint8)
    alpha_busy[80:180, 90:170] = 255
    alpha_busy[0:30, 0:256] = 255  # second blob touching the border
    bad, _ = clean_background_score(busy, alpha_busy)
    assert bad < good - 0.3


def test_rank_puts_floor_failures_last():
    scores = [
        VariationScore(0, 0.2, 1.0, 0.3, 1.0, 0.9, False),
        VariationScore(1, 0.6, 0.5, 0.25, 0.8, 0.6, True),
        VariationScore(2, 0.7, 0.6, 0.26, 0.9, 0.7, True),
    ]
    assert rank_variations(scores) == [2, 1, 0]


def test_clean_alpha_drops_islands_and_framing():
    alpha = np.zeros((200, 200), np.uint8)
    alpha[50:150, 50:150] = 255
    alpha[5:8, 5:8] = 255
    cleaned = clean_alpha(alpha)
    assert cleaned[6, 6] == 0 and cleaned[100, 100] == 255

    rgba = Image.fromarray(np.dstack([np.full((200, 200, 3), 128, np.uint8), cleaned]))
    framed = np.asarray(resize_foreground(rgba, 0.8))
    assert framed.shape[0] == framed.shape[1]
    ys, _ = np.nonzero(framed[..., 3])
    assert abs((ys.max() - ys.min() + 1) / framed.shape[0] - 0.8) < 0.02
