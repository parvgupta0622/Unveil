import cv2
import numpy as np
import pytest
from PIL import Image

from tests.conftest import draw_outline
from unveil.preprocess import SketchError, preprocess_sketch
from unveil.scoring import mask_iou


def _disk(size=512, margin=0.08):
    img = np.zeros((size, size), np.uint8)
    r = int(size * (1 - 2 * margin) / 2)
    cv2.circle(img, (size // 2, size // 2), r, 1, -1)
    return img > 0


def test_circle_outline_fills_to_disk(circle_sketch):
    prep = preprocess_sketch(circle_sketch)
    assert not prep.filled_input
    assert mask_iou(prep.silhouette, _disk()) > 0.9
    control = np.asarray(prep.control_image("precise"))
    assert control.shape == (512, 512, 3)
    assert control.mean() < 40  # white lines on black


def test_freeform_strokes_thicker_than_precise(circle_sketch):
    prep = preprocess_sketch(circle_sketch)
    precise = (np.asarray(prep.control_image("precise"))[..., 0] > 0).sum()
    freeform = (np.asarray(prep.control_image("freeform"))[..., 0] > 0).sum()
    assert freeform > 1.5 * precise


def test_empty_canvas_raises():
    with pytest.raises(SketchError):
        preprocess_sketch(Image.new("RGB", (512, 512), "white"))
    with pytest.raises(SketchError):
        preprocess_sketch(Image.new("RGBA", (512, 512), (0, 0, 0, 0)))  # untouched transparent canvas


def test_transparent_canvas_with_strokes():
    rgba = np.zeros((512, 512, 4), np.uint8)
    cv2.rectangle(rgba, (100, 100), (400, 400), (0, 0, 0, 255), 4)
    prep = preprocess_sketch(Image.fromarray(rgba))
    assert prep.silhouette.mean() > 0.5


def test_inverted_sketch(circle_sketch):
    inverted = Image.fromarray(255 - np.asarray(circle_sketch))
    assert mask_iou(preprocess_sketch(inverted).silhouette, _disk()) > 0.9


def test_photo_with_uneven_lighting(circle_sketch):
    arr = np.asarray(circle_sketch).astype(np.float32)
    gradient = np.linspace(0.55, 1.0, arr.shape[1])[None, :]  # shadow across the paper
    noisy = arr * gradient + np.random.default_rng(0).normal(0, 4, arr.shape)
    prep = preprocess_sketch(Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8)))
    assert mask_iou(prep.silhouette, _disk()) > 0.85


def test_small_off_centre_sketch_is_recentred():
    img = np.full((800, 1200), 255, np.uint8)
    cv2.rectangle(img, (900, 50), (1000, 250), 0, 3)
    prep = preprocess_sketch(Image.fromarray(img))
    ys, xs = np.nonzero(prep.silhouette)
    assert abs(xs.mean() - 256) < 10 and abs(ys.mean() - 256) < 10
    assert ys.max() - ys.min() > 400  # scaled up to fill the canvas


def test_filled_shape_input():
    img = np.full((512, 512), 255, np.uint8)
    cv2.circle(img, (256, 256), 150, 0, -1)
    prep = preprocess_sketch(Image.fromarray(img))
    assert prep.filled_input
    assert mask_iou(prep.silhouette, _disk()) > 0.9


def test_open_strokes_use_hull():
    img = np.full((512, 512), 255, np.uint8)
    cv2.line(img, (100, 100), (400, 100), 0, 3)
    cv2.line(img, (100, 400), (400, 400), 0, 3)
    prep = preprocess_sketch(Image.fromarray(img))
    assert prep.silhouette.mean() > 0.4  # hull of two parallel lines is a filled rectangle


def test_specks_removed():
    img = np.asarray(draw_outline("square")).copy()
    rng = np.random.default_rng(1)
    for y, x in rng.integers(0, 512, (200, 2)):
        img[y, x] = 0  # salt noise
    prep = preprocess_sketch(Image.fromarray(img))
    square = np.zeros((512, 512), bool)
    square[40:472, 40:472] = True
    assert mask_iou(prep.silhouette, square) > 0.9
