import os
from volume_estimator_module import VolumeEstimator


def main():
    """
    Main function to demonstrate volume estimation on a pair of food images.

    Required inputs:
      - Two images of the same food from different angles (side + top view)
      - Two corresponding .npy segmentation masks (from Module 1 / SAM)
      - pixel_length_mm calibration value (from Module 3 / thumb detection)
    """

    # ── Configuration ─────────────────────────────────────────────────────
    IMAGE1_PATH      = "input/images/food_side.jpg"    # side view image
    IMAGE2_PATH      = "input/images/food_top.jpg"     # top view image
    MASK1_PATH       = "masks/food_side_mask.npy"      # .npy mask for side view
    MASK2_PATH       = "masks/food_top_mask.npy"       # .npy mask for top view
    FOOD_TYPE        = "rice"                           # food class name (for shape correction)
    PIXEL_LENGTH_MM  = 0.5                              # mm per pixel — from Module 3
    VOXEL_SIZE_MM    = 5.0                              # voxel side length in mm

    # ── Initialize estimator ──────────────────────────────────────────────
    print("Initializing Volume Estimator...")
    estimator = VolumeEstimator(
        pixel_length_mm = PIXEL_LENGTH_MM,
        voxel_size_mm   = VOXEL_SIZE_MM,
    )
    print("✓ Estimator initialized successfully")

    # ── Check files exist ─────────────────────────────────────────────────
    for path, label in [
        (IMAGE1_PATH, "Image 1 (side)"),
        (IMAGE2_PATH, "Image 2 (top)"),
        (MASK1_PATH,  "Mask 1 (side)"),
        (MASK2_PATH,  "Mask 2 (top)"),
    ]:
        if not os.path.exists(path):
            print(f"Error: {label} not found at {path}")
            return

    # ── Run estimation ────────────────────────────────────────────────────
    print(f"\nEstimating volume for: {FOOD_TYPE}")
    print(f"  Image 1 : {IMAGE1_PATH}")
    print(f"  Image 2 : {IMAGE2_PATH}")
    print(f"  Mask 1  : {MASK1_PATH}")
    print(f"  Mask 2  : {MASK2_PATH}")

    result = estimator.estimate_by_paths(
        image1_path = IMAGE1_PATH,
        image2_path = IMAGE2_PATH,
        mask1_path  = MASK1_PATH,
        mask2_path  = MASK2_PATH,
        food_type   = FOOD_TYPE,
    )

    # ── Display results ───────────────────────────────────────────────────
    if result:
        print("\nVolume Estimation Results:")
        print(f"  - Food Type        : {result['food_type']}")
        print(f"  - Volume           : {result['volume_ml']:.2f} mL  ({result['volume_cm3']:.4f} cm³)")
        print(f"  - Volume (mm³)     : {result['volume_mm3']:.2f} mm³")
        print(f"  - Occupied voxels  : {result['num_voxels']}")
        print(f"  - Point cloud size : {result['num_points']}")
        print(f"  - Shape correction : {result['shape_correction']}")
    else:
        print("\n❌ Estimation failed — check image/mask inputs.")


if __name__ == "__main__":
    main()
