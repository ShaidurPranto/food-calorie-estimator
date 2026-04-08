from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "input" / "images"
MASKS_DIR = PROJECT_ROOT / "masks"

TOP_IMAGE = "image_beguni_top.jpg"
SIDE_IMAGE = "image_beguni_side.jpg"
TOP_PREFIX = "input_1_top_mask_"
SIDE_PREFIX = "input_1_side_mask_"


def load_mask(path: Path) -> np.ndarray:
    """Load and normalize a numpy segmentation mask to uint8 0/255."""
    data = np.load(str(path), allow_pickle=True)
    mask = np.asarray(data)

    if mask.ndim > 2:
        mask = np.squeeze(mask)

    return (mask > 0).astype(np.uint8) * 255


def get_image_shape(image_name: str) -> tuple[int, int]:
    image_path = IMAGES_DIR / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Could not load image: {image_path}")

    with Image.open(image_path) as img:
        w, h = img.size

    return (h, w)


def get_mask_files(prefix: str) -> list[Path]:
    return sorted(p for p in MASKS_DIR.glob(f"{prefix}*.npy"))


def describe_mask(mask_path: Path, image_shape: tuple[int, int]) -> dict:
    mask = load_mask(mask_path)
    h, w = mask.shape
    ih, iw = image_shape
    non_zero = int(np.count_nonzero(mask))

    return {
        "name": mask_path.name,
        "shape": (h, w),
        "pixels": non_zero,
        "matches_image_size": (h, w) == (ih, iw),
    }


def check_view(image_name: str, prefix: str) -> tuple[list[dict], tuple[int, int]]:
    image_shape = get_image_shape(image_name)
    mask_files = get_mask_files(prefix)

    if not mask_files:
        print(f"No mask files found for prefix: {prefix}")
        return [], image_shape

    print(f"\n{image_name} -> {len(mask_files)} mask(s)")
    rows: list[dict] = []
    for p in mask_files:
        row = describe_mask(p, image_shape)
        rows.append(row)
        print(
            f"  {row['name']}: shape={row['shape']}, "
            f"pixels={row['pixels']}, same_as_image={row['matches_image_size']}"
        )

    return rows, image_shape


def main() -> None:
    print("Checking NPY masks for two-view volume estimation")
    print(f"Project root: {PROJECT_ROOT}")

    top_rows, top_shape = check_view(TOP_IMAGE, TOP_PREFIX)
    side_rows, side_shape = check_view(SIDE_IMAGE, SIDE_PREFIX)

    print("\nSummary")
    print(f"  Top image shape : {top_shape}")
    print(f"  Side image shape: {side_shape}")
    print(f"  Top masks count : {len(top_rows)}")
    print(f"  Side masks count: {len(side_rows)}")

    matched_pairs = min(len(top_rows), len(side_rows))
    print(f"  Matched pairs for volume calculation: {matched_pairs}")

    if len(top_rows) != len(side_rows):
        print("\nWarning: Unequal mask counts between top and side views.")
        print("Only min(top_masks, side_masks) items can be paired automatically.")

    print("\nSuggested notebook values:")
    print(f"  IMAGE1_FILENAME = '{TOP_IMAGE}'")
    print(f"  IMAGE2_FILENAME = '{SIDE_IMAGE}'")

    top_names = [r["name"] for r in top_rows]
    side_names = [r["name"] for r in side_rows]
    print(f"  MASKS1_FILENAMES = {top_names}")
    print(f"  MASKS2_FILENAMES = {side_names}")
    print("  CLASS_IDS = None")


if __name__ == "__main__":
    main()
