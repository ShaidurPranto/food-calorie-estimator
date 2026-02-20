import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import timm

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "food_data_english"
MODEL_PATH = "saved_models/model_1_vit_segment_aware.pth"
NUM_CLASSES = 19  # Must match training
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# CHECK IF MODEL EXISTS
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please run training_pipeline_1.py first.")
    exit(1)

# -----------------------------
# TRANSFORM
# -----------------------------
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# DATASET & LOADER
# -----------------------------
test_dir = os.path.join(DATA_DIR, "test")
if not os.path.exists(test_dir):
    print(f"Error: Test directory not found at {test_dir}")
    exit(1)

test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4, pin_memory=True
)

print(f"Test classes: {len(test_dataset.classes)}")

# -----------------------------
# LOAD MODEL
# -----------------------------
print(f"Loading model from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Re-create the model structure
model = timm.create_model(
    # "vit_base_patch16_224_in21k",
    "vit_small_patch16_224",
    pretrained=False, # Verify if num_classes matches to avoid error, usually fine if loading state dict
    num_classes=NUM_CLASSES
)

# Load state dict
# The saved checkpoint has "model_state_dict" key based on training script
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    # In case it was saved differently (e.g. just model.state_dict())
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

# -----------------------------
# EVALUATION LOOP
# -----------------------------
if __name__ == "__main__":
    correct = 0
    total = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
