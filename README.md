# Faster R-CNN for Remote Sensing Image Analysis

## Project Overview
This project focuses on automated, accurate, and scalable object detection and segmentation in satellite imagery. Using **Faster R-CNN** for multi-class object detection and **DeepLabv3+** for forest cover segmentation, we aim to classify and calculate the proportion of forest cover in satellite images. The project leverages Kaggle’s **High-Resolution Satellite Dataset** for model training and evaluation.

### Key Features:
- **Multi-class Object Detection**: Detect various objects in satellite imagery, including roads, buildings, vegetation, water bodies, and more.
- **Forest Cover Calculation**: Using segmentation to calculate the forest cover percentage from satellite images.
- **Preprocessing and Data Augmentation**: Implemented various techniques for image preprocessing and augmentation to enhance model robustness.

## Dataset
The project uses **Kaggle’s High-Resolution Satellite Dataset**, which contains high-resolution RGB images for training, validation, and testing.

- **Source**: [DeepGlobe Land Cover Classification Dataset](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)
- **Classes**: 
    - Agricultural Sector
    - Terminal
    - Beach
    - City
    - Desert
    - Forest
    - Road
    - Lake
    - Mountain
    - Car Parking
    - Port
    - Train
    - Domestic
    - River

## Installation

### 1. Clone the repository
git clone https://github.com/yourusername/faster-rcnn-remote-sensing.git
cd faster-rcnn-remote-sensing

### 2. Install dependencies
Ensure you have Python 3.6 or above installed, and then install the required packages:

pip install -r requirements.txt

Preprocessing and Augmentation
The following techniques are applied to ensure high-quality data for training:

Image Preprocessing:

Corrupted image removal using hashing algorithms (MD5).

Noise reduction using median and Gaussian filters.

Image quality assessment and resizing.

Color space conversion and normalization.

Histogram equalization and adaptive histogram equalization for contrast enhancement.

Edge enhancement and blurring.

Data Augmentation:

Random rotations, flipping, cropping, scaling, shearing, and translation.

Perspective and elastic transformations.

Simulated weather effects and lighting variations.

Model
Faster R-CNN
Backbone: ResNet-50 with Feature Pyramid Network (FPN) for efficient object detection.

Optimizer: Adam with a learning rate of 0.001, tuned using ReduceLROnPlateau.

Pre-trained weights: IMAGENET1K_V1 for faster convergence.

DeepLabv3+ for Forest Segmentation
Backbone: ResNet-50.

Model: DeepLabv3+ for segmentation tasks.

Objective: Segment forest and non-forest areas from satellite imagery to calculate forest cover percentage.

Results
Accuracy: The Faster R-CNN model achieved an accuracy of 94.17% in object detection tasks.

Forest Cover Results:

Processed 2019.png: Forest Cover = 16.26%

Processed 2022.png: Forest Cover = 75.57%

Processed 2023.png: Forest Cover = 49.60%

Processed 2024.png: Forest Cover = 61.72%

Results are saved in forest_cover_results.csv for further analysis.
