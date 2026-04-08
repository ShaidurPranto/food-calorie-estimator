"""
filter_food_segment.py
======================
Identifies and extracts the 'food' segment from a set of SAM-generated
binary masks by matching them against YOLO polygon annotations.

Workflow
--------
1. Load all SAM .npy masks for a given image.
2. Load food-class (class 0) YOLO polygon annotations from the paired .txt file.
3. Convert each polygon to a binary raster mask at full image resolution.
4. Compute IoU between every SAM mask and the combined food annotation mask.
5. The SAM mask with the highest IoU is labelled as 'food'.
6. Save the food-only image (transparent background) and the food mask PNG.

Directory layout expected
-------------------------
<project>/
  input/
    images/          <- original JPG images
    image_*_1.txt    <- YOLO polygon annotation files
  masks/
    input_*_mask_*.npy  <- SAM binary masks (bool, shape H×W)
  output/            <- created automatically
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "input" / "images"
ANNOTATIONS_DIR = PROJECT_ROOT / "input"
MASKS_DIR = PROJECT_ROOT / "masks"
OUTPUT_DIR = PROJECT_ROOT / "output"

# YOLO class id that corresponds to food
FOOD_CLASS_ID = 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_sam_mask(path: Path, target_wh: tuple[int, int] | None = None) -> np.ndarray:
    """Load a SAM .npy mask as a bool array, optionally resizing to (width, height)."""
    data = np.load(str(path), allow_pickle=True)
    mask = np.asarray(data)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    mask = (mask > 0)

    if target_wh is not None:
        w, h = target_wh
        if mask.shape != (h, w):
            pil_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            pil_mask = pil_mask.resize((w, h), Image.NEAREST)
            mask = np.array(pil_mask) > 0

    return mask


def yolo_polygon_to_mask(coords: list[float], width: int, height: int) -> np.ndarray:
    """
    Convert a flat list of normalized YOLO polygon coordinates
    [x1, y1, x2, y2, ...] to a binary bool mask of shape (height, width).
    """
    points = [
        (coords[i] * width, coords[i + 1] * height)
        for i in range(0, len(coords), 2)
    ]
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(points, fill=255)
    return np.array(canvas) > 0


def load_food_annotation_mask(
    annotation_path: Path, width: int, height: int
) -> np.ndarray:
    """
    Parse a YOLO polygon .txt file and return the union of all food-class
    (class 0) polygons as a single binary mask of shape (height, width).
    """
    combined = np.zeros((height, width), dtype=bool)

    with open(annotation_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if int(parts[0]) == FOOD_CLASS_ID:
                coords = [float(v) for v in parts[1:]]
                combined |= yolo_polygon_to_mask(coords, width, height)

    if not combined.any():
        raise ValueError(
            f"No food annotations (class {FOOD_CLASS_ID}) found in {annotation_path}"
        )
    return combined


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over Union of two bool masks."""
    intersection = (a & b).sum()
    union = (a | b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


# ── Core logic ─────────────────────────────────────────────────────────────────

def find_food_mask(
    sam_mask_paths: list[Path],
    annotation_path: Path,
    image_width: int,
    image_height: int,
) -> tuple[Path, np.ndarray, float]:
    """
    Compare each SAM mask against the food annotation and return the one
    with the best IoU along with that score.

    Returns
    -------
    (best_mask_path, food_mask_bool_array, iou_score)
    """
    food_annotation = load_food_annotation_mask(
        annotation_path, image_width, image_height
    )

    best_path: Path | None = None
    best_mask: np.ndarray | None = None
    best_iou = -1.0

    print(f"\n{'Mask file':<35} {'pixels':>10}  {'IoU':>6}")
    print("-" * 55)

    for mask_path in sam_mask_paths:
        sam_mask = load_sam_mask(mask_path, target_wh=(image_width, image_height))
        iou = compute_iou(sam_mask, food_annotation)
        marker = " <-- food" if iou > best_iou else ""

        # Update best before printing so the marker is correct
        if iou > best_iou:
            best_iou = iou
            best_path = mask_path
            best_mask = sam_mask
            marker = " <-- food"
        else:
            marker = ""

        print(f"  {mask_path.name:<33} {int(sam_mask.sum()):>10}  {iou:>6.3f}{marker}")

    # Re-print the best row marker correctly (simple re-announcement)
    print(f"\nSelected food mask : {best_path.name}  (IoU = {best_iou:.3f})")
    return best_path, best_mask, best_iou


def extract_food_region(
    image_path: Path,
    food_mask: np.ndarray,
    output_path: Path,
) -> None:
    """
    Apply the food mask to the original image.
    Non-food pixels become transparent (RGBA output).
    """
    image = Image.open(image_path).convert("RGBA")
    arr = np.array(image)
    arr[:, :, 3] = np.where(food_mask, 255, 0).astype(np.uint8)
    Image.fromarray(arr, mode="RGBA").save(output_path)


# ── Entry point ────────────────────────────────────────────────────────────────

def run(
    image_name: str,
    annotation_file: str,
    mask_prefix: str,
) -> None:
    """
    Parameters
    ----------
    image_name      : filename inside input/images/  e.g. "image_input_1_top.jpg"
    annotation_file : YOLO .txt filename inside input/  e.g. "image_beguni_1.txt"
    mask_prefix     : prefix of mask files in masks/   e.g. "input_1_top_mask_"
    """
    image_path = IMAGES_DIR / image_name
    annotation_path = ANNOTATIONS_DIR / annotation_file
    mask_paths = sorted(MASKS_DIR.glob(f"{mask_prefix}*.npy"))

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation not found: {annotation_path}")
    if not mask_paths:
        raise FileNotFoundError(f"No masks found for prefix '{mask_prefix}' in {MASKS_DIR}")

    with Image.open(image_path) as img:
        image_width, image_height = img.size  # PIL: (width, height)

    print("=" * 55)
    print(f"Image      : {image_name}  ({image_width} x {image_height})")
    print(f"Annotation : {annotation_file}")
    print(f"SAM masks  : {len(mask_paths)}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    _, food_mask, _ = find_food_mask(
        mask_paths, annotation_path, image_width, image_height
    )

    stem = Path(image_name).stem
    food_image_out = OUTPUT_DIR / f"food_only_{stem}.png"
    food_mask_out = OUTPUT_DIR / f"food_mask_{stem}.png"

    extract_food_region(image_path, food_mask, food_image_out)
    Image.fromarray(food_mask.astype(np.uint8) * 255, mode="L").save(food_mask_out)

    print(f"\nSaved food image : {food_image_out}")
    print(f"Saved food mask  : {food_mask_out}")
    print("=" * 55)


def main() -> None:
    # ── Top view ──────────────────────────────────────────────────────────────
    run(
        image_name="image_input_1_top.jpg",
        annotation_file="image_beguni_1.txt",
        mask_prefix="input_1_top_mask_",
    )

    # ── Side view ─────────────────────────────────────────────────────────────
    run(
        image_name="image_input_1_side.jpg",
        annotation_file="image_beguni_2.txt",
        mask_prefix="input_1_side_mask_",
    )


if __name__ == "__main__":
    main()
