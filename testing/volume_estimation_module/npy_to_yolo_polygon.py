"""
npy_to_yolo_polygon.py
======================
Converts SAM .npy binary masks → YOLO polygon segmentation .txt format,
exactly like image.txt:

    class_id  x1 y1  x2 y2  ...  xn yn   (normalised [0-1], space-separated)

Pipeline per mask
-----------------
1. Load the .npy bool mask and resize it to the original image resolution.
2. Extract the outer contour using matplotlib marching-squares.
3. Simplify the polygon with Ramer-Douglas-Peucker (RDP) to reduce point count.
4. Normalise pixel coordinates → [0, 1].
5. Assign class id:
     - If an annotation .txt is provided → IoU-match to decide food (class 0)
       vs other (class 1). Masks whose IoU < MIN_IOU_TO_KEEP are skipped.
     - If no annotation file → write all masks as class 0.
6. Write one line per polygon to the output .txt file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "input" / "images"
ANNOTATIONS_DIR = PROJECT_ROOT / "input"
MASKS_DIR = PROJECT_ROOT / "masks"
OUTPUT_DIR = PROJECT_ROOT / "output"

FOOD_CLASS_ID = 0
OTHER_CLASS_ID = 1
MIN_IOU_TO_KEEP = 0.03      # skip masks that don't overlap any annotation at all
RDP_EPSILON = 3.0           # pixels – higher = fewer polygon points
COORD_DECIMALS = 6          # decimal places, matching image.txt style


# ── Geometry helpers ───────────────────────────────────────────────────────────

def rdp_simplify(pts: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker polygon simplification.
    pts : (N, 2) float array of (x, y) points.
    Returns a simplified (M, 2) array.
    """
    if len(pts) < 3:
        return pts

    start, end = pts[0], pts[-1]
    seg = end - start
    seg_len = np.linalg.norm(seg) + 1e-12
    # Perpendicular distances from every point to the start→end line
    dists = np.abs(np.cross(seg, start - pts)) / seg_len

    max_idx = int(np.argmax(dists))
    if dists[max_idx] > epsilon:
        left = rdp_simplify(pts[: max_idx + 1], epsilon)
        right = rdp_simplify(pts[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.array([start, end])


_MAX_CONTOUR_DIM = 640   # downsample mask to this size before tracing contour


def mask_to_polygon(mask: np.ndarray, epsilon: float = RDP_EPSILON) -> np.ndarray | None:
    """
    Extract the largest contour from a binary bool mask and simplify it.

    mask  : (H, W) bool at full image resolution
    Returns (N, 2) float array of NORMALISED [0-1] (x, y) coords,
    or None if no contour can be found.

    Steps
    -----
    1. Downsample mask to ≤ _MAX_CONTOUR_DIM px on the longest side so that
       the contour has a manageable, evenly-spaced set of vertices.
    2. Extract the outer contour with matplotlib marching-squares.
    3. Detect and handle closed-loop paths (first pt == last pt):
       rotate to a "corner" anchor before applying RDP, then close again.
    4. Simplify with RDP (epsilon in downsampled-pixel space).
    5. Normalise by the DOWNSAMPLED dimensions — this gives the same [0-1]
       values as normalising by the original dimensions (scale cancels out).
    """
    h, w = mask.shape

    # ── 1. Downsample ──────────────────────────────────────────────────────────
    scale = min(1.0, _MAX_CONTOUR_DIM / max(h, w))
    dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
    if scale < 1.0:
        pil_ds = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        pil_ds = pil_ds.resize((dw, dh), Image.NEAREST)
        mask_ds = np.array(pil_ds) > 0
    else:
        mask_ds = mask

    # ── 2. Contour extraction ──────────────────────────────────────────────────
    fig, ax = plt.subplots()
    cs = ax.contour(mask_ds.astype(float), levels=[0.5])
    plt.close(fig)

    best: np.ndarray | None = None
    best_len = 0

    for collection in cs.collections:
        for path in collection.get_paths():
            verts = path.vertices   # (N, 2) in (col, row) = (x, y) pixel order
            if len(verts) > best_len:
                best_len = len(verts)
                best = verts

    if best is None or len(best) < 3:
        return None

    pts = best.copy()

    # ── 3. Handle closed loop ──────────────────────────────────────────────────
    # matplotlib closes contours: first pt ≈ last pt → RDP baseline is ~zero
    # → all distances are ~0 → degenerates to 2 points.
    # Fix: remove the duplicate, rotate to the "sharpest corner" as anchor,
    # append a closing copy, run RDP, then strip the closing point.
    closed = np.allclose(pts[0], pts[-1], atol=0.6)
    if closed:
        pts = pts[:-1]                              # drop duplicate closing pt
        centroid = pts.mean(axis=0)
        anchor = int(np.argmax(np.linalg.norm(pts - centroid, axis=1)))
        pts = np.roll(pts, -anchor, axis=0)         # rotate so anchor is first
        pts = np.vstack([pts, pts[0:1]])            # re-close for RDP

    # ── 4. RDP simplification (epsilon in downsampled-pixel space) ────────────
    eps_ds = epsilon * scale if scale < 1.0 else epsilon
    simplified = rdp_simplify(pts, eps_ds)

    if closed:
        simplified = simplified[:-1]                # remove closing duplicate

    if len(simplified) < 3:
        return None

    # ── 5. Normalise to [0, 1] using downsampled dimensions ───────────────────
    # norm_x = x_ds / dw  ≡  (x_ds / scale) / (dw / scale)  = x_orig / w
    norm = simplified / np.array([dw, dh], dtype=float)
    return norm


def polygon_to_yolo_line(class_id: int, pts: np.ndarray) -> str:
    """
    Format a polygon as a YOLO segmentation line.
    pts : (N, 2) normalised [0-1] (x, y) coords  (returned by mask_to_polygon)
    """
    norm_coords: list[str] = []
    for nx, ny in pts:
        norm_coords.append(f"{float(nx):.{COORD_DECIMALS}f}")
        norm_coords.append(f"{float(ny):.{COORD_DECIMALS}f}")
    return f"{class_id} " + " ".join(norm_coords)


# ── Mask helpers (shared with filter_food_segment.py) ─────────────────────────

def load_sam_mask(path: Path, target_wh: tuple[int, int] | None = None) -> np.ndarray:
    data = np.load(str(path), allow_pickle=True)
    mask = np.asarray(data)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    mask = mask > 0
    if target_wh is not None:
        w, h = target_wh
        if mask.shape != (h, w):
            pil_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            pil_mask = pil_mask.resize((w, h), Image.NEAREST)
            mask = np.array(pil_mask) > 0
    return mask


def yolo_polygon_to_mask(coords: list[float], width: int, height: int) -> np.ndarray:
    points = [(coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords), 2)]
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(points, fill=255)
    return np.array(canvas) > 0


def load_food_annotation_mask(annotation_path: Path, width: int, height: int) -> np.ndarray:
    combined = np.zeros((height, width), dtype=bool)
    with open(annotation_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) == FOOD_CLASS_ID:
                coords = [float(v) for v in parts[1:]]
                combined |= yolo_polygon_to_mask(coords, width, height)
    return combined


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = (a & b).sum()
    union = (a | b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


# ── Core conversion ────────────────────────────────────────────────────────────

def masks_to_yolo_txt(
    image_name: str,
    mask_prefix: str,
    annotation_file: str | None = None,
    output_txt_name: str | None = None,
) -> None:
    """
    Parameters
    ----------
    image_name      : e.g. "image_input_1_top.jpg"
    mask_prefix     : e.g. "input_1_top_mask_"
    annotation_file : optional YOLO .txt in input/ to determine class ids
    output_txt_name : output filename (default: same stem as image_name)
    """
    image_path = IMAGES_DIR / image_name
    mask_paths = sorted(MASKS_DIR.glob(f"{mask_prefix}*.npy"))

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_paths:
        raise FileNotFoundError(f"No masks found for prefix '{mask_prefix}'")

    with Image.open(image_path) as img:
        image_width, image_height = img.size

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = output_txt_name or Path(image_name).stem
    out_path = OUTPUT_DIR / f"{stem}.txt"

    # Optional: load food annotation to classify masks
    food_annotation: np.ndarray | None = None
    if annotation_file:
        ann_path = ANNOTATIONS_DIR / annotation_file
        if ann_path.exists():
            food_annotation = load_food_annotation_mask(ann_path, image_width, image_height)

    print(f"\n{'='*58}")
    print(f"Image  : {image_name}  ({image_width}x{image_height})")
    print(f"Masks  : {len(mask_paths)}")
    print(f"Output : {out_path.name}")
    print(f"{'='*58}")
    print(f"{'Mask':<35} {'pixels':>8}  {'IoU':>6}  {'class':>5}  {'pts':>5}")
    print("-" * 58)

    lines: list[str] = []

    for mask_path in mask_paths:
        sam_mask = load_sam_mask(mask_path, target_wh=(image_width, image_height))

        # Determine class id
        if food_annotation is not None:
            iou = compute_iou(sam_mask, food_annotation)
            if iou < MIN_IOU_TO_KEEP:
                print(f"  {mask_path.name:<33} {int(sam_mask.sum()):>8}  {iou:>6.3f}  {'skip':>5}")
                continue
            # Best-overlap mask → food (class 0), others → class 1
            # We compare against all masks to find which has highest IoU
            class_id = FOOD_CLASS_ID if iou >= 0.1 else OTHER_CLASS_ID
        else:
            iou = 0.0
            class_id = FOOD_CLASS_ID

        polygon = mask_to_polygon(sam_mask, epsilon=RDP_EPSILON)
        if polygon is None:
            print(f"  {mask_path.name:<33} {int(sam_mask.sum()):>8}  {iou:>6.3f}  {'no contour':>11}")
            continue

        line = polygon_to_yolo_line(class_id, polygon)
        lines.append(line)
        iou_str = f"{iou:.3f}" if food_annotation is not None else "  n/a"
        print(f"  {mask_path.name:<33} {int(sam_mask.sum()):>8}  {iou_str:>6}  {class_id:>5}  {len(polygon):>5}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    print(f"\nWrote {len(lines)} polygon(s) → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Top view
    masks_to_yolo_txt(
        image_name="image_input_2_top.jpg",
        mask_prefix="input_2_top_mask_",
        annotation_file="image_beguni_1.txt",
        output_txt_name="image_input_2_top",
    )

    # Side view
    masks_to_yolo_txt(
        image_name="image_input_2_side.jpg",
        mask_prefix="input_2_side_mask_",
        annotation_file="image_beguni_2.txt",
        output_txt_name="image_input_2_side",
    )


if __name__ == "__main__":
    main()
