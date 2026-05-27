# =============================================================
#         PNEUMONIA DETECTION USING PYTORCH (ResNet18)
#         Binary Classification: NORMAL vs PNEUMONIA
# =============================================================


# ── IMPORTS ───────────────────────────────────────────────────

import os
# Builds cross-platform dataset folder paths.

import copy
# Creates true independent snapshot of model weights.

import torch
from torch import nn
# torch → tensors, GPU support, autograd.
# nn    → layers, loss functions, model base class.

from torchvision import models
# Provides pretrained CNN models (ResNet, EfficientNet, etc.)

import torch.optim as optim
# Optimization algorithms — Adam, SGD, etc.

from torchvision import datasets, transforms
# datasets   → ImageFolder auto-loads class-labeled images.
# transforms → preprocessing pipeline (resize, normalize, augment).

from torch.utils.data import DataLoader
# Batches dataset and feeds it to the model during training.

import matplotlib.pyplot as plt
# Plots loss curves, accuracy curves, and image grids.

from sklearn.metrics import (
    confusion_matrix,        # TP, TN, FP, FN counts
    classification_report,   # Precision, Recall, F1 per class
    ConfusionMatrixDisplay    # Renders confusion matrix as heatmap
)

from tqdm import tqdm
# Wraps DataLoader with a live progress bar during training.


# ── DEVICE SETUP ──────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Uses GPU if available, else CPU.
# All tensors and model must be moved to this device.

print(f"Using Device: {device}")


# =============================================================
#                       DATA SETUP
# =============================================================

# ── DATASET PATHS ─────────────────────────────────────────────

dataset_path = "chest_xray"

train_dir = os.path.join(dataset_path, "train_1")
val_dir   = os.path.join(dataset_path, "val")
test_dir  = os.path.join(dataset_path, "test_1")
# Builds full paths to each split folder.
# Expected subfolders inside each: NORMAL/ and PNEUMONIA/


# ── TRANSFORMS ────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # Resizes to 224×224 — required input size for ResNet18.

    transforms.RandomHorizontalFlip(),
    # Random left-right flip for data augmentation.

    transforms.RandomRotation(10),
    # Random ±10° rotation — simulates real X-ray tilt variation.

    transforms.ToTensor(),
    # Converts PIL image to tensor, scales pixels [0,255] → [0,1].

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # Normalizes with ImageNet mean & std — matches ResNet18 pretraining.
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # No augmentation for val/test — clean images for fair evaluation.
])


# ── DATASETS ──────────────────────────────────────────────────

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_dir,   transform=test_transform)
test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transform)
# ImageFolder auto-assigns class labels from subfolder names:
#   NORMAL    → 0
#   PNEUMONIA → 1


# ── DATALOADERS ───────────────────────────────────────────────

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# shuffle=True → randomizes order every epoch, reduces bias.

val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
# shuffle=False → consistent order for reliable evaluation.


# ── CLASS NAMES ───────────────────────────────────────────────

class_names = train_dataset.classes
# Retrieves class labels: ['NORMAL', 'PNEUMONIA']

print(f"Classes: {class_names}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")


# ── VISUALIZE FIRST 5 TRAINING IMAGES ────────────────────────

images, labels = next(iter(train_loader))
# Fetches the first batch from the training loader.

plt.figure(figsize=(15, 5))

for i in range(5):

    image = images[i].permute(1, 2, 0).numpy()
    # permute(1,2,0) → converts (C,H,W) to (H,W,C) for matplotlib.

    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    # Reverses normalization so image colors display correctly.

    image = image.clip(0, 1)
    # Clamps values to [0,1] — denorm can push some pixels out of range.

    plt.subplot(1, 5, i + 1)
    plt.imshow(image)
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()


# =============================================================
#                       MODEL SETUP
# =============================================================

# ── LOAD PRETRAINED RESNET18 ──────────────────────────────────

model = models.resnet18(weights="DEFAULT")
# Loads ResNet18 pretrained on ImageNet — strong feature extractor.

for param in model.parameters():
    param.requires_grad = False
# Freezes all pretrained layers — only the new final layer will train.
# Saves compute, prevents overfitting, preserves ImageNet knowledge.

model.fc = nn.Linear(512, 2)
# Replaces original 1000-class output with 2-class output:
#   512 → input features from ResNet18's backbone
#   2   → NORMAL and PNEUMONIA

model = model.to(device)
# Moves model to GPU/CPU.

print("Model Loaded Successfully!")


# ── LOSS & OPTIMIZER ──────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
# Standard loss for classification.
# Applies Softmax internally — penalizes confident wrong predictions.

optimizer = optim.Adam(
    model.fc.parameters(),  # Only optimize the unfrozen final layer
    lr=0.001                # Learning rate for weight update steps
)


# =============================================================
#                     TRAINING FUNCTION
# =============================================================

def train_model(model, epochs):

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    # Saves a true copy of weights — updated whenever val accuracy improves.

    train_losses   = []
    val_accuracies = []
    # Store per-epoch metrics for plotting later.

    for epoch in range(epochs):

        print(f"\nEpoch [{epoch + 1}/{epochs}]")

        # ── TRAINING ──────────────────────────────────────────

        model.train()
        # Enables Dropout & BatchNorm training behavior.

        running_loss = 0.0
        loop = tqdm(train_loader)

        for images, labels in loop:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # Clears gradients from previous batch — required every step.

            outputs = model(images)
            # Forward pass → raw class scores (logits).

            loss = criterion(outputs, labels)
            # Computes loss between predictions and true labels.

            loss.backward()
            # Backpropagation — computes gradients.

            optimizer.step()
            # Updates model.fc weights using computed gradients.

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        # Average loss across all batches in the epoch.

        train_losses.append(epoch_loss)
        print(f"Train Loss: {epoch_loss:.4f}")


        # ── VALIDATION ────────────────────────────────────────

        model.eval()
        # Disables Dropout — consistent predictions during evaluation.

        correct = 0
        total   = 0

        with torch.no_grad():
            # No gradients needed — saves memory and speeds up inference.

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                # Picks the class with the highest score per image.

                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        print(f"Validation Accuracy: {accuracy:.2f}%")

        # ── SAVE BEST WEIGHTS ─────────────────────────────────

        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            # Saves weights only when accuracy improves.

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")

    model.load_state_dict(best_model_wts)
    # Restores the best performing weights before returning.

    return model, train_losses, val_accuracies


# =============================================================
#                     TRAIN & SAVE MODEL
# =============================================================

epochs = 5

model, train_losses, val_accuracies = train_model(model, epochs)
# Runs full training — returns best model + metrics lists.

torch.save(model.state_dict(), "pneumonia_resnet18.pth")
# Saves only weights (state_dict) — lightweight and portable.
# Load later with: model.load_state_dict(torch.load("pneumonia_resnet18.pth"))

print("\nModel Saved Successfully!")


# =============================================================
#                     EVALUATION (TEST SET)
# =============================================================

model.eval()

all_labels      = []
all_predictions = []
correct = 0
total   = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        # .cpu() → back to CPU | .numpy() → to NumPy array
        # Collected across all batches for full metrics below.

test_accuracy = 100 * correct / total
print(f"\nTest Accuracy: {test_accuracy:.2f}%")


# ── CLASSIFICATION REPORT ─────────────────────────────────────

print("\nClassification Report:\n")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=class_names
))
# Prints Precision, Recall, F1-Score per class.
# Better than accuracy alone — especially for imbalanced datasets.


# ── CONFUSION MATRIX ──────────────────────────────────────────

cm = confusion_matrix(all_labels, all_predictions)
# Matrix of TP, TN, FP, FN counts.

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
disp.plot()
plt.title("Confusion Matrix")
plt.show()


# =============================================================
#                     PLOT METRICS
# =============================================================

# ── TRAINING LOSS CURVE ───────────────────────────────────────

plt.figure(figsize=(8, 5))
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# Healthy trend: loss steadily decreasing ↘


# ── VALIDATION ACCURACY CURVE ─────────────────────────────────

plt.figure(figsize=(8, 5))
plt.plot(val_accuracies)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
# Healthy trend: accuracy steadily increasing ↗
# If accuracy drops while loss falls → overfitting.


# =============================================================
#                   SHOW FINAL PREDICTIONS
# =============================================================

def show_predictions(model, loader):

    model.eval()

    images, labels = next(iter(loader))
    # Fetches one batch from the given loader.

    images_device = images.to(device)

    outputs = model(images_device)
    _, predicted = torch.max(outputs, 1)
    # Gets model's predicted class for each image in the batch.

    plt.figure(figsize=(15, 5))

    for i in range(5):

        image = images[i].permute(1, 2, 0).numpy()
        image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image = image.clip(0, 1)
        # Denormalize for correct display — same as visualize step above.

        plt.subplot(1, 5, i + 1)
        plt.imshow(image)

        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]

        plt.title(f"True: {true_label}\nPred: {pred_label}")
        # Shows both ground truth and model prediction per image.
        # Green title = correct | Red title = wrong (add color if needed).

        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ── DISPLAY TEST PREDICTIONS ──────────────────────────────────

show_predictions(model, test_loader)
