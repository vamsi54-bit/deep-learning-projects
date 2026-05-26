🩺 Pneumonia Detection from Chest X-Ray Images using ResNet18
Overview

This project presents a deep learning-based approach for automated pneumonia detection from chest X-ray images using PyTorch and a pretrained ResNet model.

The system leverages transfer learning with ResNet18 to classify radiographic images into two categories:

Normal
Pneumonia

The implementation includes the complete computer vision workflow:

data preprocessing
augmentation
model training
validation
performance evaluation
prediction visualization

This project demonstrates the practical application of convolutional neural networks in medical image classification tasks.

Objective

The primary objective of this project is to develop a robust image classification model capable of identifying pneumonia-related patterns in chest X-ray scans.

The project focuses on:

transfer learning
medical imaging workflows
model evaluation techniques
deep learning model optimization
Dataset

Dataset used:

Chest X-ray Pneumonia Dataset

Dataset structure:

chest_xray/

    train/
        NORMAL/
        PNEUMONIA/

    val/
        NORMAL/
        PNEUMONIA/

    test/
        NORMAL/
        PNEUMONIA/

The dataset contains labeled chest radiographs for supervised binary classification.

Model Architecture

The project utilizes a pretrained ResNet18 architecture.

H(x)=F(x)+x

Residual connections enable deeper neural networks to train efficiently by mitigating the vanishing gradient problem.

The final fully connected classification layer was modified to support binary classification:

model.fc = nn.Linear(512, 2)
Methodology
1. Data Preprocessing

Input images are:

resized to 224×224
converted into tensors
normalized using ImageNet statistics
2. Data Augmentation

Training data augmentation techniques include:

random horizontal flipping
random rotation

These transformations improve generalization and reduce overfitting.

3. Transfer Learning

A pretrained ResNet18 model was used as the feature extractor while replacing the final classification layer for the target task.

4. Optimization
Loss Function: CrossEntropyLoss
Optimizer: Adam
Batch Size: 32
Epochs: 5
Evaluation Metrics

Model performance is evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

These metrics provide a comprehensive assessment of classification performance, particularly for medical datasets where class imbalance may exist.

Features
Transfer learning with ResNet18
Chest X-ray image classification
Automated training and validation pipeline
Confusion matrix visualization
Classification report generation
Training loss visualization
Validation accuracy plotting
Sample prediction visualization
GPU acceleration support
Technologies Used
Technology	Purpose
PyTorch	Model development
torchvision	Pretrained models and transforms
Matplotlib	Visualization
scikit-learn	Evaluation metrics
tqdm	Training progress monitoring
Project Workflow
Chest X-ray Image
        ↓
Preprocessing & Augmentation
        ↓
ResNet18 Feature Extraction
        ↓
Binary Classification
        ↓
Prediction Output
Results

The trained model is capable of learning discriminative radiographic features associated with pneumonia and performing binary classification on unseen chest X-ray images.

The project also generates:

confusion matrix visualization
classification report
prediction samples
training performance plots
Installation

Install required dependencies:

pip install torch torchvision matplotlib scikit-learn tqdm
Execution

Run the project using:

python pneumonia_detection.py
Output

The project generates:

trained model weights (pneumonia_resnet18.pth)
confusion matrix
training loss graph
validation accuracy graph
prediction visualizations
Future Enhancements

Potential future improvements include:

Grad-CAM explainability
web-based deployment
multi-class thoracic disease classification
hyperparameter optimization
real-time inference systems
Conclusion

This project demonstrates the effective use of transfer learning and convolutional neural networks for medical image classification tasks. It provides practical exposure to deep learning workflows, evaluation methodologies, and real-world computer vision applications using PyTorch.
