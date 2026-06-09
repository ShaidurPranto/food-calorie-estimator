from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def foreground_mask(image: np.ndarray) -> np.ndarray:
    """Build a binary foreground mask from alpha channel or pixel intensity."""
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        mask = (alpha > 10).astype(np.uint8) * 255
    else:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No foreground contour found.")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 25:
        raise ValueError("Foreground contour is too small for calibration.")
    return contour


def finger_axes_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    """Return (length_px, width_px) from oriented minimum-area rectangle."""
    contour = largest_contour(mask)
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if w <= 0 or h <= 0:
        raise ValueError("Invalid contour geometry for finger dimensions.")
    length_px = float(max(w, h))
    width_px = float(min(w, h))
    return length_px, width_px


def tight_crop_foreground(image: np.ndarray, pad: int = 8) -> np.ndarray:
    """Crop to foreground for more stable visual embeddings."""
    mask = foreground_mask(image)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image

    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, image.shape[1] - 1)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad, image.shape[0] - 1)
    return image[y0 : y1 + 1, x0 : x1 + 1]
