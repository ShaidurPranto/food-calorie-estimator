import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "dataset-v3"
OLD_MODEL_PATH = "saved_models/model_v2_vit_segment_aware.pth"
NEW_MODEL_PATH = "saved_models/model_v3_vit_segment_aware.pth"
NEW_LABELS_PATH = "saved_models/labels_v3.txt"

BATCH_SIZE = 8
EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("saved_models", exist_ok=True)

# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0
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
# DATASET
# ============================================================

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

full_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train")
)

NUM_CLASSES = len(full_dataset.classes)

print(f"Found {NUM_CLASSES} classes")
print("Class order used by the model:")

for idx, name in enumerate(full_dataset.classes):
    print(f"{idx}: {name}")

# Save labels in EXACT training order
with open(NEW_LABELS_PATH, "w", encoding="utf-8") as f:
    for cls in full_dataset.classes:
        f.write(cls + "\n")

print(f"\nSaved labels to {NEW_LABELS_PATH}")

# Train/Val split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_subset, val_subset = torch.utils.data.random_split(
    full_dataset,
    [train_size, val_size]
)

train_dataset = TransformSubset(train_subset, train_transform)
val_dataset = TransformSubset(val_subset, val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ============================================================
# LOAD OLD MODEL 
# ============================================================

checkpoint = torch.load(OLD_MODEL_PATH, map_location="cpu")

old_num_classes = len(checkpoint["class_to_idx"])

print(f"\nOld model classes: {old_num_classes}")

old_model = timm.create_model(
    "vit_base_patch16_224_in21k",
    pretrained=False,
    num_classes=old_num_classes
)

old_model.load_state_dict(checkpoint["model_state_dict"])

# ============================================================
# CREATE NEW MODEL 
# ============================================================

model = timm.create_model(
    "vit_base_patch16_224_in21k",
    pretrained=False,
    num_classes=NUM_CLASSES
)

# Copy all backbone weights
model.patch_embed.load_state_dict(old_model.patch_embed.state_dict())
model.blocks.load_state_dict(old_model.blocks.state_dict())
model.norm.load_state_dict(old_model.norm.state_dict())

# ============================================================
# COPY OLD CLASSIFIER WEIGHTS USING CLASS NAMES
# ============================================================

old_class_to_idx = checkpoint["class_to_idx"]
new_class_to_idx = full_dataset.class_to_idx

with torch.no_grad():
    for class_name, old_idx in old_class_to_idx.items():
        new_idx = new_class_to_idx[class_name]

        model.head.weight[new_idx] = old_model.head.weight[old_idx]
        model.head.bias[new_idx] = old_model.head.bias[old_idx]

print("Transferred old classifier weights successfully!")

model.to(DEVICE)

# ============================================================
# FREEZE BACKBONE FOR FIRST FEW EPOCHS
# ============================================================

for param in model.patch_embed.parameters():
    param.requires_grad = False

for param in model.blocks.parameters():
    param.requires_grad = False

for param in model.norm.parameters():
    param.requires_grad = False

# ============================================================
# LOSS & OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.head.parameters(),
    lr=1e-4,
    weight_decay=0.05
)

# ============================================================
# TRAINING FUNCTIONS
# ============================================================

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

def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total

# ============================================================
# TRAIN
# ============================================================

best_acc = 0

for epoch in range(EPOCHS):

    # Unfreeze backbone after 3 epochs
    if epoch == 3:
        print("\nUnfreezing backbone...")

        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(
            [
                {"params": model.patch_embed.parameters(), "lr": 1e-6},
                {"params": model.blocks.parameters(), "lr": 1e-6},
                {"params": model.norm.parameters(), "lr": 1e-6},
                {"params": model.head.parameters(), "lr": 1e-4},
            ],
            weight_decay=0.05
        )

    train_loss = train_one_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_acc:
        best_acc = val_acc

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": full_dataset.class_to_idx
            },
            NEW_MODEL_PATH
        )

        print(f"Saved best model to {NEW_MODEL_PATH}")

print(f"\nTraining complete!")
print(f"Best validation accuracy: {best_acc:.2f}%")
print(f"Model saved at: {NEW_MODEL_PATH}")
print(f"Labels saved at: {NEW_LABELS_PATH}")