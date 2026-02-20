import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "food_data_english"
NUM_CLASSES = 19          # change if needed
BATCH_SIZE = 8
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("saved_models", exist_ok=True)
MODEL_SAVE_PATH = "saved_models/model_1_vit_segment_aware.pth"

# -----------------------------
# 1. SEGMENTATION-AWARE AUGMENTATION (TRAIN)
# -----------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # ← MUST come before RandomErasing
    transforms.RandomErasing(
        p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0
    ),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # also fixing normalization
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 2. STANDARD VALIDATION TRANSFORM
# -----------------------------
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# 3. DATASETS & LOADERS
# -----------------------------
# Define a helper to apply transforms to a Subset
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Load the full dataset from the 'train' folder
full_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"))

# Split into train (80%) and validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Apply respective transforms
# Note: full_dataset has no transform applied by default if none passed, but we passed none above?
# Wait, ImageFolder constructor defaults transform=None.
# But inside the subset, the underlying dataset (full_dataset) is accessed.
# If we wrap the subset with our transforms, it works perfectly.

train_dataset = TransformSubset(train_subset, transform=train_transform)
val_dataset = TransformSubset(val_subset, transform=val_transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4, pin_memory=True
)

print(f"Classes: {full_dataset.classes}")

# -----------------------------
# 4. MODEL (ViT ImageNet-21k)
# -----------------------------
model = timm.create_model(
    # "vit_base_patch16_224_in21k",
    "vit_small_patch16_224",
    pretrained=True,
    num_classes=NUM_CLASSES
)
model.to(DEVICE)

# -----------------------------
# 5. LOSS & OPTIMIZER (DIFFERENTIAL LR)
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {"params": model.patch_embed.parameters(), "lr": 1e-5},
        {"params": model.blocks.parameters(), "lr": 1e-5},
        {"params": model.head.parameters(), "lr": 1e-4},
    ],
    weight_decay=0.05
)

# -----------------------------
# 6. TRAINING LOOP
# -----------------------------
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# -----------------------------
# 7. EVALUATION
# -----------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total

# -----------------------------
# 8. TRAIN
# -----------------------------
if __name__ == "__main__":
    best_acc = 0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader)
        val_acc = evaluate(model, val_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": full_dataset.class_to_idx
                },
                MODEL_SAVE_PATH
            )

    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to {MODEL_SAVE_PATH}")
