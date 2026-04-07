import os
from torchvision import datasets

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "food_data_english/train"  # point to your train folder
OUTPUT_FILE = "saved_models/labels.txt"

os.makedirs("saved_models", exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = datasets.ImageFolder(root=DATA_DIR)

# class_to_idx example:
# {'apple_pie': 0, 'burger': 1, ...}

class_to_idx = dataset.class_to_idx

# -----------------------------
# SORT BY INDEX (IMPORTANT!)
# -----------------------------
# We want index -> class_name order
idx_to_class = {v: k for k, v in class_to_idx.items()}

# -----------------------------
# WRITE TO FILE
# -----------------------------
with open(OUTPUT_FILE, "w") as f:
    for i in range(len(idx_to_class)):
        f.write(idx_to_class[i] + "\n")

print(f"labels.txt saved to: {OUTPUT_FILE}")
print("Classes in order:")
for i in range(len(idx_to_class)):
    print(f"{i}: {idx_to_class[i]}")