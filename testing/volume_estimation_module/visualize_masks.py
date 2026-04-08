from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "input" / "images"
MASKS_DIR = PROJECT_ROOT / "masks"
OUTPUT_DIR = PROJECT_ROOT / "output"

TOP_IMAGE = "image_input_1_top.jpg"
SIDE_IMAGE = "image_input_1_side.jpg"

TOP_MASKS = ["input_1_top_mask_000.npy", "input_1_top_mask_001.npy", "input_1_top_mask_002.npy"]
SIDE_MASKS = ["input_1_side_mask_000.npy", "input_1_side_mask_001.npy", "input_1_side_mask_002.npy", "input_1_side_mask_003.npy"]

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # R, G, B, Y


def load_mask(path: Path) -> np.ndarray:
    """Load and normalize a numpy segmentation mask."""
    data = np.load(str(path), allow_pickle=True)
    mask = np.asarray(data)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    return (mask > 0).astype(np.uint8)


def resize_mask(mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize mask to match image dimensions. target_size should be (width, height)."""
    if mask.shape != target_size[::-1]:  # Reverse since numpy is (h, w) but PIL is (w, h)
        mask_img = Image.fromarray(mask * 255, mode='L')
        mask_img = mask_img.resize(target_size, Image.NEAREST)
        return np.array(mask_img) > 0
    return mask


def draw_mask_overlay(image_path: Path, mask_paths: list[Path], output_path: Path, label: str) -> None:
    """Load image, overlay all masks with different colors, save result."""
    print(f"\nProcessing {label}...")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    h, w = image.size[1], image.size[0]  # PIL returns (width, height)
    
    print(f"  Image shape: {image.size} → (h={h}, w={w})")
    
    # Create overlay
    overlay = image.copy()
    overlay_array = np.array(overlay)
    
    # Load and overlay each mask with different color
    for idx, mask_path in enumerate(mask_paths):
        mask = load_mask(mask_path)
        mask = resize_mask(mask, (w, h))  # Pass (width, height) for PIL
        
        color = COLORS[idx % len(COLORS)]
        pixel_count = int(np.count_nonzero(mask))
        
        print(f"  Mask {idx}: {mask_path.name} ({pixel_count} pixels) -> Color {color}")
        
        # Create colored overlay for this mask
        mask_indices = np.where(mask)
        overlay_array[mask_indices] = np.array(color, dtype=np.uint8)
    
    # Save overlay
    overlay_result = Image.fromarray(overlay_array, mode='RGB')
    overlay_result.save(output_path)
    print(f"  ✅ Saved: {output_path}")


def main() -> None:
    print("=" * 60)
    print("VISUALIZING TOP AND SIDE MASKS")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Process top view
    top_image_path = IMAGES_DIR / TOP_IMAGE
    top_mask_paths = [MASKS_DIR / f for f in TOP_MASKS]
    top_output = OUTPUT_DIR / "visualize_top_masks.png"
    
    if all(p.exists() for p in [top_image_path] + top_mask_paths):
        draw_mask_overlay(top_image_path, top_mask_paths, top_output, "TOP VIEW")
    else:
        print("❌ Top image or masks not found")
    
    # Process side view
    side_image_path = IMAGES_DIR / SIDE_IMAGE
    side_mask_paths = [MASKS_DIR / f for f in SIDE_MASKS]
    side_output = OUTPUT_DIR / "visualize_side_masks.png"
    
    if all(p.exists() for p in [side_image_path] + side_mask_paths):
        draw_mask_overlay(side_image_path, side_mask_paths, side_output, "SIDE VIEW")
    else:
        print("❌ Side image or masks not found")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nOpen these files in your image viewer:")
    print(f"  {top_output}")
    print(f"  {side_output}")


if __name__ == "__main__":
    main()
