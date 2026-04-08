import os
import cv2
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


DATASET_BASE = "/kaggle/input/datasets/ifty3110/segmentation-module-checkpoints-config"

class SegmentationModule:
    """
    A segmentation module for food image segmentation using SAM2.
    """

    def __init__(self):
        self.checkpoint    = os.path.join(DATASET_BASE, "checkpoints/checkpoints/sam2_hiera_large.pt")
        self.model_config  = os.path.join(DATASET_BASE, "configs/configs/sam2/sam2_hiera_l.yaml")
        self.device        = "cuda"

        self.points_per_side        = 24
        self.points_per_batch       = 32
        self.pred_iou_thresh        = 0.8
        self.stability_score_thresh = 0.92
        self.box_nms_thresh         = 0.75
        self.min_mask_region_area   = 2000
        self.dedup_iou_threshold    = 0.85
        self.min_area_ratio         = 0.30

        self.mask_generator = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"SAM2 checkpoint not found at {self.checkpoint}")

        print(f"Loading SAM2 on {self.device}...")
        sam2 = build_sam2(self.model_config, self.checkpoint,
                          device=self.device, apply_postprocessing=True)

        return SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=self.points_per_side,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            box_nms_thresh=self.box_nms_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

    def _load_image_rgb(self, input):
        if isinstance(input, str):
            if not os.path.exists(input):
                raise FileNotFoundError(f"Image not found at {input}")
            image = cv2.imdecode(np.fromfile(input, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to read image: {input}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        elif isinstance(input, Image.Image):
            return np.array(input.convert("RGB"))

        else:
            raise TypeError("Input must be a file path (str) or a PIL.Image object")

    def _generate_masks(self, image_rgb):
        masks = self.mask_generator.generate(image_rgb)
        masks = self._deduplicate_masks(masks)

        if masks:
            max_area = max(m["area"] for m in masks)
            masks = [m for m in masks if m["area"] >= self.min_area_ratio * max_area]

        return masks

    def _deduplicate_masks(self, masks):
        masks = sorted(masks, key=lambda m: m["area"], reverse=True)
        kept = []
        for candidate in masks:
            c_mask = candidate["segmentation"]
            is_duplicate = False
            for kept_mask in kept:
                k_mask = kept_mask["segmentation"]
                intersection = np.logical_and(c_mask, k_mask).sum()
                union = np.logical_or(c_mask, k_mask).sum()
                if union > 0 and (intersection / union) > self.dedup_iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(candidate)
        return kept

    def set_image_from_path(self, image_path):
        self._cached_image_rgb = self._load_image_rgb(image_path)
        self._cached_masks = None

    def set_image(self, image):
        self._cached_image_rgb = self._load_image_rgb(image)
        self._cached_masks = None

    def _get_cached_masks(self, input=None):
        if input is not None:
            image_rgb = self._load_image_rgb(input)
            return image_rgb, self._generate_masks(image_rgb)

        if not hasattr(self, "_cached_image_rgb"):
            raise RuntimeError("No image loaded. Call set_image() or set_image_from_path() first.")

        if self._cached_masks is None:
            self._cached_masks = self._generate_masks(self._cached_image_rgb)

        return self._cached_image_rgb, self._cached_masks

    def get_coordinates(self, input=None):
        _, masks = self._get_cached_masks(input)
        return [
            {
                "segment_id":      i,
                "bbox":            mask["bbox"],
                "area":            mask["area"],
                "predicted_iou":   round(mask["predicted_iou"], 4),
                "stability_score": round(mask["stability_score"], 4),
            }
            for i, mask in enumerate(masks)
        ]

    def get_masks(self, input=None):
        _, masks = self._get_cached_masks(input)
        return [
            {
                "segment_id": i,
                "mask": mask["segmentation"].astype(bool),
                "area": mask["area"],
            }
            for i, mask in enumerate(masks)
        ]

    def get_segments(self, input=None):
        image_rgb, masks = self._get_cached_masks(input)
        results = []

        for i, mask in enumerate(masks):
            mask_bool = mask["segmentation"].astype(bool)

            rgba = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3] = image_rgb
            rgba[..., 3] = (mask_bool * 255).astype(np.uint8)

            ys, xs = np.where(mask_bool)
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            cropped = rgba[y0:y1, x0:x1]

            results.append({
                "segment_id": i,
                "image": Image.fromarray(cropped, mode="RGBA"),
                "bbox": mask["bbox"],
                "area": mask["area"],
            })

        return results

    def clear_cache(self):
        self._cached_image_rgb = None
        self._cached_masks = None
        torch.cuda.empty_cache()