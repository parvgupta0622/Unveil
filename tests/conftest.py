import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def draw_outline(shape: str = "circle", size: int = 512, width: int = 4) -> Image.Image:
    img = np.full((size, size), 255, np.uint8)
    c = size // 2
    if shape == "circle":
        cv2.circle(img, (c, c), size // 3, 0, width)
    elif shape == "square":
        cv2.rectangle(img, (c - size // 4, c - size // 4), (c + size // 4, c + size // 4), 0, width)
    elif shape == "L":
        pts = np.array([[150, 100], [250, 100], [250, 350], [400, 350], [400, 430], [150, 430]], np.int32)
        cv2.polylines(img, [pts], True, 0, width)
    return Image.fromarray(img)


@pytest.fixture
def circle_sketch():
    return draw_outline("circle")
