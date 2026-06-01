# ExoHunter AI: Deep Learning-Based Exoplanet Detection Using NASA Kepler Light Curves

## Overview

ExoHunter AI is a deep learning project that leverages real NASA Kepler telescope observations to detect potential exoplanet signatures from stellar light curves. The project processes astronomical time-series data and compares the performance of Long Short-Term Memory (LSTM) and Bidirectional Long Short-Term Memory (BiLSTM) neural networks for exoplanet detection.

Unlike traditional machine learning projects that rely on static datasets, ExoHunter AI dynamically acquires real observational data from NASA archives using the Lightkurve library, creating an end-to-end astrophysics and deep learning pipeline.

---

## Objectives

* Acquire real stellar light curve data from NASA's Kepler mission.
* Preprocess and normalize astronomical time-series observations.
* Convert raw flux measurements into fixed-length sequences suitable for deep learning.
* Develop and compare LSTM and BiLSTM architectures for exoplanet detection.
* Evaluate model performance using standard classification metrics.
* Demonstrate the application of deep learning in astrophysical research.

---

## Dataset

### Source

NASA Kepler Mission Archive

### Data Type

Stellar Light Curves

### Features

* Host Star Name
* Stellar Brightness (Flux)
* Observation Cadence
* Time-Series Measurements

### Preprocessing Steps

1. Download Kepler light curves using Lightkurve.
2. Remove missing observations.
3. Interpolate irregular sequences.
4. Resample all light curves to 300 timesteps.
5. Normalize flux values.
6. Convert processed sequences into PyTorch tensors.

---

## Project Pipeline

NASA Exoplanet Archive
↓
Kepler Host Star Selection
↓
Light Curve Acquisition
↓
Data Cleaning
↓
Interpolation & Resampling
↓
Normalization
↓
Tensor Conversion
↓
LSTM Training
↓
BiLSTM Training
↓
Performance Evaluation
↓
Model Comparison

---

## Model Architectures

### LSTM Model

* Input Size: 1
* Hidden Size: 64
* Number of Layers: 2
* Dropout: 0.3
* Fully Connected Layers: 64 → 32 → 1
* Activation: Sigmoid

### BiLSTM Model

* Input Size: 1
* Hidden Size: 64
* Number of Layers: 2
* Bidirectional Processing
* Dropout: 0.3
* Fully Connected Layers: 128 → 32 → 1
* Activation: Sigmoid

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* SciPy
* Matplotlib
* Scikit-Learn
* Lightkurve
* NASA Kepler Archive

---

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## Sample Input

A normalized stellar light curve consisting of 300 timesteps:

Shape:
(300, 1)

Example:

[1.002, 0.998, 0.995, 1.001, 0.997, ...]

---

## Sample Output

Prediction: Exoplanet Detected

Confidence Score: 0.91

or

Prediction: No Exoplanet Detected

Confidence Score: 0.87

---

## Results

The project compares the predictive capabilities of LSTM and BiLSTM architectures on real astronomical observations. The final analysis highlights the strengths and limitations of both models for exoplanet transit detection tasks.

---

## Future Improvements

* Transformer-based time-series models
* Automated transit detection visualization
* Multi-mission support (Kepler, K2, TESS)
* Explainable AI techniques for astrophysical interpretation
* Web-based prediction dashboard

---

## Author

Vamsi

Deep Learning | Data Science | Astronomy AI

ExoHunter AI demonstrates the integration of modern deep learning techniques with real-world astronomical datasets to explore one of humanity's most exciting scientific challenges: discovering planets beyond our solar system.
