# ─────────────────────────────────────────────
# Standard Library / Warning Suppression
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")  # Suppress all runtime warnings (e.g. convergence, deprecation)

# ─────────────────────────────────────────────
# Numerical & Data Manipulation
# ─────────────────────────────────────────────
import numpy as np          # Array operations, linear algebra, math utilities
import pandas as pd         # DataFrames for tabular data loading and manipulation

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
import matplotlib.pyplot as plt  # Plotting: light curves, loss curves, confusion matrices

# ─────────────────────────────────────────────
# Astronomy / Time-Series
# ─────────────────────────────────────────────
import lightkurve as lk              # NASA Kepler/TESS light curve downloading and processing
from scipy import interpolate        # Spline/interpolation tools for filling gaps in light curves

# ─────────────────────────────────────────────
# Deep Learning (PyTorch)
# ─────────────────────────────────────────────
import torch                                      # Core PyTorch tensor library and autograd
import torch.nn as nn                             # Neural network layers, loss functions, activations
from torch.utils.data import Dataset, DataLoader  # Custom dataset class and mini-batch data loader

# ─────────────────────────────────────────────
# Machine Learning (Scikit-learn)
# ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split  # Split data into train/validation/test sets

from sklearn.metrics import (
    accuracy_score,        # Overall fraction of correct predictions
    precision_score,       # TP / (TP + FP) — how precise positive predictions are
    recall_score,          # TP / (TP + FN) — how many positives were caught
    f1_score,              # Harmonic mean of precision and recall
    confusion_matrix,      # Matrix of actual vs predicted class counts
    classification_report  # Full per-class precision, recall, F1 summary
)
