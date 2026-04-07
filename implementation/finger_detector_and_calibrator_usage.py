from __future__ import annotations
# pyright: reportMissingImports=false

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finger_detector_and_calibrator_module import (
    FingerDetectorAndCalibrator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Usage script for FingerDetectorAndCalibrator module."
    )
    parser.add_argument(
        "--segmented_dir",
        type=Path,
        default=ROOT / "data" / "segmented_outputs",
        help="Directory containing segmented images.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=ROOT / "models" / "finger_detector.joblib",
        help="Trained detector model path.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "models" / "module_usage_result.json",
        help="Path to save full result JSON.",
    )
    parser.add_argument(
        "--output_txt",
        type=Path,
        default=ROOT / "models" / "module_usage_cm_per_pixel.txt",
        help="Path to save only cm_per_pixel value.",
    )
    parser.add_argument(
        "--allow_low_confidence",
        action="store_true",
        help="Use best-scored image even if below threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    module = FingerDetectorAndCalibrator()
    module.load_model(args.model_path)

    detection = module.detect_finger(
        segmented_dir=args.segmented_dir,
        allow_low_confidence=args.allow_low_confidence,
    )
    calibration = module.calibrate_finger_image(detection["chosen_finger_image"])

    result = {
        **detection,
        "calibration": calibration.to_dict(),
        "finger_length_cm": module.finger_length_cm,
        "finger_width_cm": module.finger_width_cm,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)

    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.output_txt.write_text(
        f"{result['calibration']['cm_per_pixel']:.12f}\n",
        encoding="utf-8",
    )

    print("Module usage summary")
    print(f"Chosen finger image: {result['chosen_finger_image']}")
    print(f"Chosen score: {result['chosen_image_score']:.6f}")
    print(f"Threshold: {result['detector_threshold']:.6f}")
    print(f"cm_per_pixel: {result['calibration']['cm_per_pixel']:.12f}")
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved TXT: {args.output_txt}")


if __name__ == "__main__":
    main()
