from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

from .image_utils import finger_axes_from_mask, foreground_mask, read_image


@dataclass
class CalibrationResult:
    image_path: str
    finger_length_px: float
    finger_width_px: float
    finger_length_cm: float
    finger_width_cm: float
    cm_per_pixel_from_length: float
    cm_per_pixel_from_width: float
    cm_per_pixel: float
    pixel_per_cm: float
    cm2_per_pixel2: float
    agreement_ratio: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


def calibrate_from_finger_mask(
    *,
    image_path: str | Path,
    finger_length_cm: float,
    finger_width_cm: float,
) -> CalibrationResult:
    if finger_length_cm <= 0 or finger_width_cm <= 0:
        raise ValueError("finger_length_cm and finger_width_cm must be positive.")

    image = read_image(image_path)
    mask = foreground_mask(image)
    length_px, width_px = finger_axes_from_mask(mask)

    cm_per_pixel_l = finger_length_cm / length_px
    cm_per_pixel_w = finger_width_cm / width_px

    cm_per_pixel = (cm_per_pixel_l + cm_per_pixel_w) / 2.0
    pixel_per_cm = 1.0 / cm_per_pixel
    cm2_per_pixel2 = cm_per_pixel * cm_per_pixel
    agreement_ratio = max(cm_per_pixel_l, cm_per_pixel_w) / min(cm_per_pixel_l, cm_per_pixel_w)

    return CalibrationResult(
        image_path=str(image_path),
        finger_length_px=length_px,
        finger_width_px=width_px,
        finger_length_cm=finger_length_cm,
        finger_width_cm=finger_width_cm,
        cm_per_pixel_from_length=cm_per_pixel_l,
        cm_per_pixel_from_width=cm_per_pixel_w,
        cm_per_pixel=cm_per_pixel,
        pixel_per_cm=pixel_per_cm,
        cm2_per_pixel2=cm2_per_pixel2,
        agreement_ratio=agreement_ratio,
    )


def calibrate_from_image(
    image_path: str | Path,
    finger_length_cm: float,
    finger_width_cm: float,
) -> CalibrationResult:
    return calibrate_from_finger_mask(
        image_path=image_path,
        finger_length_cm=finger_length_cm,
        finger_width_cm=finger_width_cm,
    )
