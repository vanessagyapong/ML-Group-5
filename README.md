#  Pneumonia Detection

## Introduction

This project aims to develop a machine learning model to classify chest X-ray images as either normal or pneumonia-infected. The goal is to assist medical professionals in making faster and more effective diagnoses. Traditional diagnostic methods rely on radiologists, which can be time-consuming.

## Methodology

### Dataset

- **Chest X-ray images** categorized into normal and pneumonia-infected classes.
- Data pre-processed to maintain consistent quality and dimensions.
- Dataset split into **training (80%)**, **validation (10%)**, and **testing (10%)** sets.

### Data Preprocessing

- **Resizing**: Images resized to 224x224 pixels.
- **Normalization**: Pixel values scaled to [0,1].
- **Splitting**: Training, validation, and testing sets created.

### Model Architecture

- **CNNs**: Used as the primary architecture.
- **Pretrained Models**: VGG16 and ResNet50 applied via transfer learning.
- **Custom CNN Model**: Designed with multiple convolutional layers, batch normalization, and dropout layers.

### Training Process

- **Loss Function**: Categorical cross-entropy.
- **Optimizer**: Adam.
- **Performance Metrics**: Accuracy, precision, recall, F1-score, AUC-ROC.

## Result of Analysis

### Data Processing

- Missing values handled.
- Data normalized and split.

### Model Performance

- **Metrics**: Accuracy, precision, recall, F1-score, ROC-AUC.
- **Confusion Matrix**

### Discussion

- **Accuracy**: 88% on the test set.
- **F1-Scores**: 0.86-0.89 for both classes.

