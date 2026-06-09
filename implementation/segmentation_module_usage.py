import os
import json
import numpy as np
from PIL import Image
from segmentation_module import SegmentationModule

# =============================
# CONFIGURATION
# =============================
INPUT_DIR  = r"D:\academics\buet_cse_undergrad\3-2\CAPSTONE_PROJECT_REPO\SAMPLE_INPUT"
OUTPUT_DIR = r"D:\academics\buet_cse_undergrad\3-2\CAPSTONE_PROJECT_REPO\SAMPLE_OUTPUT"

COORDS_DIR   = os.path.join(OUTPUT_DIR, "coordinates")
MASKS_DIR    = os.path.join(OUTPUT_DIR, "masks")
SEGMENTS_DIR = os.path.join(OUTPUT_DIR, "segments")

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

# =============================
# SETUP
# =============================
os.makedirs(COORDS_DIR,   exist_ok=True)
os.makedirs(MASKS_DIR,    exist_ok=True)
os.makedirs(SEGMENTS_DIR, exist_ok=True)

seg = SegmentationModule()

# =============================
# PROCESSING LOOP
# =============================
image_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(SUPPORTED_EXTENSIONS)
]

print(f"Found {len(image_files)} images in input folder.\n")

for filename in image_files:
    image_path = os.path.join(INPUT_DIR, filename)
    base_name  = os.path.splitext(filename)[0]

    print(f"Processing: {filename}")

    # ── Per-image subfolders ──────────────────────────────────────────
    img_coords_dir   = os.path.join(COORDS_DIR,   base_name)
    img_masks_dir    = os.path.join(MASKS_DIR,    base_name)
    img_segments_dir = os.path.join(SEGMENTS_DIR, base_name)

    os.makedirs(img_coords_dir,   exist_ok=True)
    os.makedirs(img_masks_dir,    exist_ok=True)
    os.makedirs(img_segments_dir, exist_ok=True)

    # ── Load image once, reuse cache for all three calls ─────────────
    seg.set_image_from_path(image_path)

    # ── 1. COORDINATES → JSON files ──────────────────────────────────
    coordinates = seg.get_coordinates()
    for coord in coordinates:
        sid       = coord["segment_id"]
        out_path  = os.path.join(img_coords_dir, f"{base_name}_coord_{sid:03d}.json")
        with open(out_path, "w") as f:
            json.dump(coord, f, indent=4)

    print(f"  ✔ Coordinates saved  ({len(coordinates)} segments)")

    # ── 2. MASKS → NPY files ─────────────────────────────────────────
    masks = seg.get_masks()
    for mask_data in masks:
        sid      = mask_data["segment_id"]
        out_path = os.path.join(img_masks_dir, f"{base_name}_mask_{sid:03d}.npy")
        np.save(out_path, mask_data["mask"])

    print(f"  ✔ Masks saved        ({len(masks)} segments)")

    # ── 3. SEGMENTS → PNG files ───────────────────────────────────────
    segments = seg.get_segments()
    for segment in segments:
        sid      = segment["segment_id"]
        out_path = os.path.join(img_segments_dir, f"{base_name}_segment_{sid:03d}.png")
        segment["image"].save(out_path)

    print(f"  ✔ Segments saved     ({len(segments)} segments)")

    # ── Clear cache before next image ────────────────────────────────
    seg.clear_cache()
    print()

print("All done!")

# =============================
# OUTPUT STRUCTURE
# =============================
# SAMPLE_OUTPUT/
# ├── coordinates/
# │   └── image1/
# │       ├── image1_coord_000.json
# │       ├── image1_coord_001.json
# │       └── ...
# ├── masks/
# │   └── image1/
# │       ├── image1_mask_000.npy
# │       ├── image1_mask_001.npy
# │       └── ...
# └── segments/
#     └── image1/
#         ├── image1_segment_000.png
#         ├── image1_segment_001.png
#         └── ...