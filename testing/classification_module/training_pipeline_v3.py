import os
import random
from copy import deepcopy

import torch
import torch.nn as nn
import timm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset


# ============================================================
# CONFIGURATION
# ============================================================

# Dataset root
# Structure:
#
# DATASET/
#    train/
#        class1/
#        class2/
#        ...
#
DATA_DIR = "food_data"

# Previous checkpoint.
# Set to None if training from ImageNet for the first time.
CHECKPOINT_PATH = "saved_models/model_latest.pth"

# Output model
OUTPUT_MODEL = "saved_models/model_latest.pth"

# Output labels
OUTPUT_LABELS = "saved_models/labels.txt"

# Training
EPOCHS = 20
BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("saved_models", exist_ok=True)


# ============================================================
# DATA AUGMENTATION
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value=0
    ),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET WRAPPER
# ============================================================

class TransformSubset(Dataset):

    def __init__(self, subset, transform=None):

        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):

        image, label = self.subset[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):

        return len(self.subset)


# ============================================================
# LOAD DATASET
# ============================================================

full_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train")
)

NUM_CLASSES = len(full_dataset.classes)

print("=" * 60)
print("Dataset loaded")
print("=" * 60)

print("Number of classes :", NUM_CLASSES)

print("\nClasses:\n")

for i, c in enumerate(full_dataset.classes):
    print(f"{i:2d} -> {c}")

print()

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_subset, val_subset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataset = TransformSubset(
    train_subset,
    train_transform
)

val_dataset = TransformSubset(
    val_subset,
    val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ============================================================
# LOAD / BUILD MODEL
# ============================================================

def create_model(num_classes):
    """
    Create a ViT model with the given number of output classes.
    """
    model = timm.create_model(
        "vit_base_patch16_224_in21k",
        pretrained=True,
        num_classes=num_classes
    )
    return model


# ============================================================
# UNIVERSAL CHECKPOINT LOADER
# ============================================================

if CHECKPOINT_PATH is None or not os.path.exists(CHECKPOINT_PATH):

    print("=" * 60)
    print("No checkpoint found.")
    print("Training from ImageNet pretrained weights.")
    print("=" * 60)

    model = create_model(NUM_CLASSES)

else:

    print("=" * 60)
    print("Checkpoint found.")
    print(CHECKPOINT_PATH)
    print("=" * 60)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    if "model_state_dict" in checkpoint:
        checkpoint_state = checkpoint["model_state_dict"]
    else:
        checkpoint_state = checkpoint

    if "class_to_idx" in checkpoint:
        old_class_to_idx = checkpoint["class_to_idx"]
    else:
        raise RuntimeError(
            "Checkpoint does not contain class_to_idx."
        )

    old_num_classes = len(old_class_to_idx)

    print(f"Checkpoint classes : {old_num_classes}")
    print(f"Dataset classes    : {NUM_CLASSES}")

    # ========================================================
    # CASE 1
    # SAME NUMBER OF CLASSES
    # ========================================================

    if old_num_classes == NUM_CLASSES:

        print("\nContinuing training...")

        model = create_model(NUM_CLASSES)

        model.load_state_dict(checkpoint_state)

    # ========================================================
    # CASE 2
    # NEW CLASSES ADDED
    # ========================================================

    else:

        print("\nDetected new classes.")
        print("Expanding classifier...")

        # ------------------------------
        # build old model
        # ------------------------------

        old_model = create_model(old_num_classes)

        old_model.load_state_dict(checkpoint_state)

        # ------------------------------
        # build new model
        # ------------------------------

        model = create_model(NUM_CLASSES)

        # ------------------------------
        # copy backbone
        # ------------------------------

        old_dict = old_model.state_dict()
        new_dict = model.state_dict()

        for key in old_dict:

            # skip classifier
            if key.startswith("head."):
                continue

            new_dict[key] = old_dict[key]

        model.load_state_dict(
            new_dict,
            strict=False
        )

        # ------------------------------
        # copy classifier by class name
        # ------------------------------

        new_class_to_idx = full_dataset.class_to_idx

        with torch.no_grad():

            copied = 0

            for class_name in old_class_to_idx:

                # class no longer exists
                if class_name not in new_class_to_idx:
                    continue

                old_index = old_class_to_idx[class_name]
                new_index = new_class_to_idx[class_name]

                model.head.weight[new_index] = \
                    old_model.head.weight[old_index]

                model.head.bias[new_index] = \
                    old_model.head.bias[old_index]

                copied += 1

        print(f"\nCopied classifier weights for {copied} classes.")

print("\nMoving model to device...")

model.to(DEVICE)

print("Done.")


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.patch_embed.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.blocks.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.head.parameters(),
            "lr": 1e-4
        }
    ],
    weight_decay=0.05
)

# ============================================================
# TRAIN
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# ============================================================
# VALIDATION
# ============================================================

def evaluate():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

    return 100.0 * correct / total


# ============================================================
# SAVE LABELS
# ============================================================

def save_labels():

    idx_to_class = {
        idx: cls
        for cls, idx in full_dataset.class_to_idx.items()
    }

    with open(OUTPUT_LABELS, "w", encoding="utf-8") as f:

        for i in range(len(idx_to_class)):
            f.write(idx_to_class[i] + "\n")


# ============================================================
# SAVE MODEL
# ============================================================

def save_checkpoint(best_acc):

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": full_dataset.class_to_idx,
            "best_accuracy": best_acc
        },
        OUTPUT_MODEL
    )

    save_labels()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    best_acc = 0.0

    print("=" * 70)
    print("Starting Training")
    print("=" * 70)

    for epoch in range(EPOCHS):

        train_loss = train_one_epoch()

        val_acc = evaluate()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"| Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:

            best_acc = val_acc

            save_checkpoint(best_acc)

            print("Best model saved.")

    print("=" * 70)
    print("Training Finished")
    print("=" * 70)

    print(f"Best Validation Accuracy : {best_acc:.2f}%")
    print(f"Model saved to : {OUTPUT_MODEL}")
    print(f"Labels saved to : {OUTPUT_LABELS}")