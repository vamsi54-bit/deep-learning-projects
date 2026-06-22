# NeuroSight: AI-Powered Brain Tumor Classification and Diagnostic Reporting

## Overview

NeuroSight is a deep learning-based medical imaging system designed to assist in brain tumor analysis from MRI scans. The project combines state-of-the-art computer vision models with automated report generation to provide a complete AI-assisted diagnostic workflow.

The system currently performs:

* Brain tumor classification using EfficientNet-B3
* MRI image preprocessing and augmentation
* Automated diagnostic report generation using GPT-4o
* REST API deployment through FastAPI

A future segmentation module based on U-Net has also been implemented and integrated into the architecture, with training support prepared for BraTS-style datasets.

---

## Features

### Brain Tumor Classification

Classifies MRI scans into four categories:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

### Deep Learning Architecture

#### EfficientNet-B3 Classifier

* Pretrained on ImageNet
* Fine-tuned for brain tumor classification
* Dropout regularization
* Mixed precision training support
* Cosine learning rate scheduling

#### U-Net Segmentation Module (Future Work)

* Complete U-Net architecture implemented
* Encoder-decoder design with skip connections
* Dice + BCE loss support
* Ready for BraTS2020 integration
* Not yet trained on segmentation masks

### Automated Diagnostic Reports

Generates structured clinical-style reports containing:

* Findings
* Tumor Characteristics
* Severity Assessment
* Recommended Next Steps

Reports are generated from model outputs using OpenAI GPT-4o.

### FastAPI Backend

Provides REST endpoints for:

* Health monitoring
* Tumor classification
* Diagnostic report generation

---

## Project Architecture

```text
MRI Image
    │
    ▼
Preprocessing
    │
    ▼
EfficientNet-B3
    │
    ├── Tumor Class
    └── Confidence Score
    │
    ▼
GPT-4o Report Generator
    │
    ▼
Diagnostic Report

Future Extension:
MRI Image
    │
    ▼
U-Net Segmentation
    │
    ▼
Tumor Mask + Coverage Statistics
```

## Dataset

Brain Tumor MRI Dataset:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Expected directory structure:

```text
data/
└── raw/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    │
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/NeuroSight.git
cd NeuroSight
```

### Install Dependencies

```bash
pip install torch torchvision pillow numpy matplotlib openai fastapi uvicorn python-multipart scikit-learn
```

### Configure OpenAI API Key

Linux/macOS:

```bash
export OPENAI_API_KEY="your_api_key"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

---

## Training

### Train Classifier

```bash
python neurosight.py --mode train_cls
```

Output:

```text
checkpoints/classifier.pth
```

### Train Segmentation Model

```bash
python neurosight.py --mode train_seg
```

Note: Requires BraTS2020 image-mask datasets.

---

## Inference

Run prediction on a single MRI image:

```bash
python neurosight.py --mode infer --image path/to/mri.jpg
```

Outputs:

* Predicted tumor class
* Confidence score
* Diagnostic report
* Visualization image

Generated file:

```text
neurosight_output.png
```

---

## API Deployment

Start FastAPI server:

```bash
python neurosight.py --mode api
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

### Health Endpoint

```http
GET /health
```

### Prediction Endpoint

```http
POST /predict
```

### Report Endpoint

```http
POST /report
```

---

## Model Details

| Component         | Model           |
| ----------------- | --------------- |
| Classification    | EfficientNet-B3 |
| Segmentation      | U-Net           |
| Report Generation | GPT-4o          |
| Backend           | FastAPI         |
| Framework         | PyTorch         |

---

## Current Limitations

* Segmentation model architecture is implemented but not yet trained.
* Reports are generated from model outputs only.
* Not intended for clinical diagnosis.
* Requires validation by qualified medical professionals.

---

## Future Work

* BraTS2020 segmentation training
* Multi-modal MRI support (T1, T1CE, T2, FLAIR)
* Explainable AI visualizations (Grad-CAM)
* Docker deployment
* Cloud inference pipeline
* Hospital PACS integration
* Multi-language report generation

---

## Disclaimer

NeuroSight is an academic and research project intended for educational purposes only. The system does not replace professional medical diagnosis, treatment planning, or radiological assessment. All outputs must be reviewed and validated by licensed healthcare professionals before any clinical use.

---

## Author

Vamsi

Student Research Project – Medical Imaging and Artificial Intelligence

2026
