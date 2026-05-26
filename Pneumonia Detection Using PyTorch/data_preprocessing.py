# =============================================================
#         PNEUMONIA DETECTION - DATA PREPROCESSING
# =============================================================


# ── DATASET PATHS ─────────────────────────────────────────────

dataset_path = "chest_xray"
# Root folder of the dataset downloaded from Kaggle.

train_dir = os.path.join(dataset_path, "train_1")
val_dir   = os.path.join(dataset_path, "val")
test_dir  = os.path.join(dataset_path, "test_1")
# os.path.join builds the full folder path for each split.
# Expected structure inside each folder:
#   NORMAL/     → normal chest X-rays
#   PNEUMONIA/  → infected chest X-rays


# ── TRANSFORMS ────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # Resize all images to 224×224 — required input size for
    # pretrained models like ResNet, EfficientNet.

    transforms.RandomHorizontalFlip(),
    # Randomly flips the image left-right during training.
    # Adds variety so the model doesn't memorize orientation.

    transforms.RandomRotation(10),
    # Randomly rotates image up to ±10 degrees.
    # Simulates slight tilt variation in real X-ray captures.

    transforms.ToTensor(),
    # Converts PIL image to PyTorch tensor and scales
    # pixel values from [0, 255] → [0.0, 1.0].

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # Normalizes using ImageNet mean & std.
    # These are the exact values the pretrained model was
    # trained on — using them keeps the weights valid.
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # No augmentation for val/test — only resize + normalize.
    # We evaluate on clean, unmodified images for fair results.
])


# ── DATASETS ──────────────────────────────────────────────────

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
# Loads training images with augmentation transforms.

val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
# Loads validation images (no augmentation).

test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
# Loads test images (no augmentation).

# ImageFolder automatically assigns class labels based on subfolder names:
#   NORMAL    → 0
#   PNEUMONIA → 1

print("Classes:", train_dataset.classes)
# Output: ['NORMAL', 'PNEUMONIA']

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")


# ── DATALOADERS ───────────────────────────────────────────────

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# shuffle=True → randomizes order every epoch to reduce bias.

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# shuffle=False → consistent order for reliable evaluation.

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Same as val — no shuffle needed during final testing.

# batch_size=32 → model sees 32 images at a time.
# Balances memory usage and training stability.
