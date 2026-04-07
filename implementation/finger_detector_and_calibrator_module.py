from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from helper_files.calibration import CalibrationResult, calibrate_from_image
from helper_files.modeling import FingerOneClassDetector


class FingerDetectorAndCalibrator:

    def __init__(self, *, finger_length_cm: float = 6.0, finger_width_cm: float = 1.5) -> None:
        if finger_length_cm <= 0 or finger_width_cm <= 0:
            raise ValueError("finger_length_cm and finger_width_cm must be positive.")
        self.finger_length_cm = float(finger_length_cm)
        self.finger_width_cm = float(finger_width_cm)
        self.detector: FingerOneClassDetector | None = None

    def load_model(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
    ) -> FingerOneClassDetector:
        """Load and store the trained finger detector model."""
        self.detector = FingerOneClassDetector.load(model_path, device=device)
        return self.detector

    def _require_detector(self) -> FingerOneClassDetector:
        if self.detector is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        return self.detector

    def detect_finger(
        self,
        segmented_dir: str | Path,
        *,
        allow_low_confidence: bool = False,
    ) -> Dict[str, Any]:
        """Detect the most likely finger image from segmented outputs."""
        detector = self._require_detector()

        image_paths = detector.list_images(segmented_dir)
        if not image_paths:
            raise ValueError(f"No images found in segmented_dir={segmented_dir}")

        scores: List[tuple[str, float]] = detector.batch_scores(image_paths)
        best_path, best_score = scores[0]
        threshold = float(detector.threshold)

        if best_score < threshold and not allow_low_confidence:
            raise RuntimeError(
                "No segmented image passed threshold. "
                f"best_score={best_score:.6f}, threshold={threshold:.6f}"
            )

        return {
            "chosen_finger_image": best_path,
            "chosen_image_score": best_score,
            "detector_threshold": threshold,
            "all_scores_desc": [{"path": p, "score": s} for p, s in scores],
        }

    def calibrate_finger_image(self, finger_image_path: str | Path) -> CalibrationResult:
        """Calibrate pixel scale from a chosen finger image."""
        return calibrate_from_image(
            image_path=finger_image_path,
            finger_length_cm=self.finger_length_cm,
            finger_width_cm=self.finger_width_cm,
        )

    def detect_and_calibrate(
        self,
        segmented_dir: str | Path,
        *,
        allow_low_confidence: bool = False,
    ) -> Dict[str, Any]:
        """Run full flow: detect finger image, then calibrate it."""
        detection = self.detect_finger(
            segmented_dir=segmented_dir,
            allow_low_confidence=allow_low_confidence,
        )
        calibration = self.calibrate_finger_image(detection["chosen_finger_image"])

        result: Dict[str, Any] = {
            **detection,
            "calibration": calibration.to_dict(),
            "finger_length_cm": self.finger_length_cm,
            "finger_width_cm": self.finger_width_cm,
        }
        return result


def run_module(
    *,
    model_path: str | Path,
    segmented_dir: str | Path,
    finger_length_cm: float = 6.0,
    finger_width_cm: float = 1.5,
    allow_low_confidence: bool = False,
) -> Dict[str, Any]:
    """Functional helper for one-shot usage of the module wrapper."""
    module = FingerDetectorAndCalibrator(
        finger_length_cm=finger_length_cm,
        finger_width_cm=finger_width_cm,
    )
    module.load_model(model_path)
    return module.detect_and_calibrate(
        segmented_dir=segmented_dir,
        allow_low_confidence=allow_low_confidence,
    )
