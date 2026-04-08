# Food Calorie Estimator

A computer vision pipeline that estimates food volume and calorie content from images. The system takes top-view and side-view photographs of a meal, segments individual food items, classifies them, estimates their 3D volume using a voxel-grid approach, and maps the result to nutritional data.

## Pipeline Overview

```
Top Image  ──┐
             ├──▶ [Module 1] Segmentation (SAM2)  ──▶  Instance Masks
Side Image ──┘
                         │
                         ▼
              [Module 2] Food Filter  ──▶  Discard non-food segments
                         │
                         ▼
              [Module 3] Finger Detector & Calibrator  ──▶  pixel_length_mm
                         │
                         ▼
              [Module 4] Volume Estimator (Voxel Grid)  ──▶  Volume (mL)
                         │
                         ▼
              [Module 5] Food Classifier (ViT)  ──▶  Food label + confidence
                         │
                         ▼
                  Nutrition Estimation
```

## Modules

### Module 1 — Segmentation (`segmentation_module.py`)
Uses **SAM2** (Segment Anything Model 2) to automatically generate instance masks from an image. Applies IoU-based deduplication and area-ratio filtering to keep only meaningful food-region masks.

- Input: image path or PIL Image
- Output: binary masks (`.npy`), bounding boxes, cropped RGBA segments

### Module 2 — Food Filter (`filterization_module.py`)
Classifies each segmented crop as **food** or **not food** using a pretrained `csatv2` model from HuggingFace (`mrdbourke/food-not-food-classifier-csatv2-v1`). Filters out background segments (plates, table, hands, etc.) before further processing.

- Input: folder of segmented images
- Output: binary array (1 = food, 0 = not food), optionally copies food-only segments to output folder

### Module 3 — Finger Detector & Calibrator (`finger_detector_and_calibrator_module.py`)
Detects a **finger** in the segmented outputs and uses its known physical size to compute `pixel_length_mm` — the real-world scale of one pixel. This calibration value drives accurate volume estimation.

- Input: directory of segmented images, known finger dimensions (default: 6 cm × 1.5 cm)
- Output: `pixel_length_mm` (calibration factor)

### Module 4 — Volume Estimator (`volume_estimator_module.py`)
Estimates the 3D volume of each food item using a **voxel-grid pipeline**:
1. Preprocessing masks (morphological closing, hole-filling)
2. Building a 3D point cloud via SIFT feature matching + triangulation across two views
3. Outlier removal (statistical k-NN based)
4. Voxelization to count occupied voxels
5. Applying food-type-aware **shape correction factors**

- Input: two images + two masks (side view & top view), `pixel_length_mm`, food type
- Output: volume in mm³, cm³, mL, and L

Supported shape correction factors:

| Food Type | Factor |
|-----------|--------|
| Rice, Soup, Salad | 1.00 |
| Apple, Orange, Meatball | 0.52 |
| Egg | 0.50 |
| Burger | 0.65 |
| Sandwich | 0.60 |
| Pizza, Flatbread, Naan | 0.20–0.25 |
| Cake | 0.70 |

### Module 5 — Food Classifier (`classification_module.py`)
Classifies food items using a fine-tuned **Vision Transformer** (`vit_small_patch16_224` via `timm`). Trained on 19 Bangladeshi food categories.

- Input: PIL Image or image path
- Output: class name, class index, confidence score

Supported food classes:
> Bakorkhani, Beguni, Biriyani, Chickpeas, Egg Omelette, Fuchka, Haleem, Hilsha Fish, Kabab, Kacha Golla, Kala Bhuna, Khichuri, Mashed Potato, Morog Polao, Nehari, Porota, Roshgolla, Roshmalai, Yogurt

## Project Structure

```
food-calorie-estimator/
├── implementation/
│   ├── segmentation_module.py              # Module 1: SAM2 segmentation
│   ├── filterization_module.py             # Module 2: food/non-food filter
│   ├── finger_detector_and_calibrator_module.py  # Module 3: pixel calibration
│   ├── volume_estimator_module.py          # Module 4: voxel-grid volume estimation
│   ├── classification_module.py            # Module 5: ViT food classifier
│   ├── helper_files/
│   │   ├── calibration.py
│   │   ├── image_utils.py
│   │   └── modeling.py
│   └── meta/
│       └── labels.txt                      # 19 food class labels
├── testing/
│   ├── segmentation_module/
│   ├── filterization_module/
│   ├── classification_module/
│   └── volume_estimation_module/
├── designing/
│   └── system_architecture.txt
└── studied_papers/
```

## Dependencies

- Python 3.10+
- PyTorch
- `timm`
- `segment-anything-2` (SAM2)
- OpenCV (`cv2`)
- NumPy, SciPy
- Pillow
- `huggingface_hub`, `safetensors`
- Matplotlib

## Usage

### Segmentation

```python
from segmentation_module import SegmentationModule

seg = SegmentationModule()
seg.set_image_from_path("food.jpg")

masks    = seg.get_masks()       # list of binary masks
segments = seg.get_segments()    # list of cropped RGBA PIL images
coords   = seg.get_coordinates() # bounding boxes + scores
```

### Food Filter

```python
from filterization_module import FoodFilter

ff = FoodFilter()
result = ff.is_food("segment_001.png")          # True / False
array  = ff.get_food_array("segments/")         # np.array of 0s and 1s
```

### Calibration

```python
from finger_detector_and_calibrator_module import FingerDetectorAndCalibrator

fdc = FingerDetectorAndCalibrator(finger_length_cm=6.0, finger_width_cm=1.5)
fdc.load_model("saved_models/finger_detector.pt")
result = fdc.detect_and_calibrate("segments/")
pixel_length_mm = result["calibration"]["pixel_length_mm"]
```

### Volume Estimation

```python
from volume_estimator_module import VolumeEstimator

estimator = VolumeEstimator(pixel_length_mm=0.5)
result = estimator.estimate_by_paths(
    image1_path = "images/food_side.jpg",
    image2_path = "images/food_top.jpg",
    mask1_path  = "masks/food_side_mask.npy",
    mask2_path  = "masks/food_top_mask.npy",
    food_type   = "rice",
)
print(result["volume_ml"], "mL")
```

### Food Classification

```python
from classification_module import FoodClassifier

clf = FoodClassifier()
result = clf.classify_by_path("food_crop.jpg")
print(result["class_name"], result["confidence"])
```

## Datasets

All datasets and supporting files used in this project are hosted on Kaggle:

**Kaggle Collection:** [capstone-food-calorie-estimator](https://www.kaggle.com/work?group=capstone-food-calorie-estimator)

| Dataset | Used For |
|---------|----------|
| Bangladeshi food images (19 classes) | Training the ViT food classifier (Module 5) |
| Segmented food crops with masks | Training / testing segmentation pipeline (Module 1) |
| Finger reference images | Training the finger detector (Module 3) |
| Two-view food images (side + top) | Testing volume estimation (Module 4) |

> Download the required datasets from the Kaggle collection above and place them in the appropriate `testing/` or `implementation/` subdirectories before running the modules.

## Model Files Required

Place the following in the `implementation/` directory:

| File | Description |
|------|-------------|
| `saved_models/model_1_vit_segment_aware.pth` | Fine-tuned ViT food classifier |
| `checkpoints/sam2_hiera_large.pt` | SAM2 large checkpoint |
| `configs/sam2/sam2_hiera_l.yaml` | SAM2 model config |
| Finger detector model (`.pt`) | YOLOv8/one-class detector for finger |
