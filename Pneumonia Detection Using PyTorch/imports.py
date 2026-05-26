# =============================================================
#         PNEUMONIA DETECTION - REQUIRED LIBRARIES
# =============================================================

# ── STANDARD LIBRARY ─────────────────────────────────────────

import os
# Used to interact with the file system.
# Helps in building dataset folder paths dynamically
# (e.g., os.path.join('data', 'train')) so the code
# works on any OS (Windows / Linux / Mac) without
# hardcoding path separators.

import copy
# Used to save the BEST model weights during training.
# torch model weights are mutable objects, so a normal
# assignment (best = model) just copies the reference.
# copy.deepcopy(model.state_dict()) creates a true
# independent snapshot that won't change as training continues.


# ── PYTORCH CORE ─────────────────────────────────────────────

import torch
# The main PyTorch library.
# Provides tensors (multi-dimensional arrays that run on GPU),
# autograd for automatic gradient computation, and the
# entire deep learning ecosystem used in this project.

from torch import nn
# Neural Network module.
# Contains all building blocks: layers (Linear, Conv2d),
# activation functions (ReLU), loss functions (CrossEntropyLoss),
# and the base nn.Module class that every custom model inherits.

from torchvision.models import get_model
# Factory function to load any pretrained torchvision model
# (ResNet, VGG, EfficientNet, etc.) by name as a string.
# Used here to load a pretrained CNN backbone for
# transfer learning instead of training from scratch.

import torch.optim as optim
# Optimization algorithms module.
# Provides optimizers like Adam, SGD, RMSprop that update
# model weights using gradients computed during backpropagation.


# ── TORCHVISION ───────────────────────────────────────────────

from torchvision import datasets, transforms
# datasets : Provides ImageFolder, which auto-loads images
#            organized in class-named subdirectories
#            (e.g., data/train/NORMAL/, data/train/PNEUMONIA/).
#
# transforms: Image preprocessing pipeline — Resize, CenterCrop,
#             RandomHorizontalFlip, ToTensor, Normalize, etc.
#             Applied before feeding images into the model.

from torch.utils.data import DataLoader
# Wraps a Dataset and handles batching, shuffling, and
# parallel data loading (num_workers).
# Feeds mini-batches to the model during training & evaluation
# instead of loading all images into memory at once.


# ── VISUALIZATION ─────────────────────────────────────────────

import matplotlib.pyplot as plt
# General-purpose plotting library.
# Used to plot training/validation loss & accuracy curves
# across epochs so you can visually track model learning
# and spot overfitting or underfitting early.


# ── EVALUATION METRICS ───────────────────────────────────────

from sklearn.metrics import (
    confusion_matrix,        # Matrix showing TP, TN, FP, FN counts
    classification_report,   # Per-class Precision, Recall, F1-Score
    ConfusionMatrixDisplay    # Plots the confusion matrix as a heatmap
)
# Scikit-learn metrics used AFTER training to evaluate
# how well the model distinguishes NORMAL vs PNEUMONIA.
# These give a much clearer picture than accuracy alone,
# especially since the dataset is class-imbalanced.


# ── PROGRESS BAR ──────────────────────────────────────────────

from tqdm import tqdm
# Wraps any iterable (like a DataLoader) and displays
# a live progress bar in the terminal during training.
# Shows current batch, speed (it/s), and ETA —
# makes long training loops easier to monitor.


# =============================================================
#   DEVICE SETUP  (add this right after imports in main code)
# =============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Automatically uses GPU if available, else falls back to CPU.
# All tensors and the model must be moved to this device
# using .to(device) for consistent computation.
