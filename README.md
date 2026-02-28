# 🖐 Hand Gesture Classification using MediaPipe Landmarks (HaGRID)

## Project Overview

This project aims to build a machine learning system capable of classifying hand gestures using landmark coordinates extracted from the **HaGRID (Hand Gesture Recognition Image Dataset)** via MediaPipe.

The input data consists of 3D hand landmark coordinates (x, y, z) representing 21 keypoints per detected hand.  
The output is a trained classification model that predicts the corresponding gesture class.
This project follows a structured ML workflow including data preprocessing, model training, evaluation and experiment tracking using MLflow.

---

## Dataset Description

- Dataset: HaGRID (Hand Gesture Recognition Image Dataset)
- 21 hand landmarks per sample
- Each landmark contains:
  - x coordinate
  - y coordinate
  - z coordinate
- Total numerical features: **63**
- Number of gesture classes: **18**

### Preprocessing Strategy

To ensure scale and position invariance:

- Recenter all (x, y) coordinates so the wrist landmark becomes the origin.
- Normalize (x, y) coordinates by dividing by the middle finger tip position.
- The z-coordinate is left unchanged as it is already normalized by MediaPipe.

---

##  Project Structure

```
Hand-Gesture-Recognition-HaGRID-Dataset/

Hand-Gesture-Recognition-HaGRID-Dataset/
├── Data/                   # Dataset files
├── Notebooks/              # Source code: Visualization, Preprocessing, Models, Evaluation
├── MLflow Screenshots/     # MLflow UI screenshots (research experiments)
├── mlruns/                 # MLflow experiment tracking (research branch)
├── MLflow Notebooks/       # Notebooks related to MLflow experiments
├── README.md               # This file
└── Requirements.txt        # Project dependencies

```

## 🤖 Machine Learning Models

At least three classification models will be implemented and compared including:

- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- (Optional) Gradient Boosting / XGBoost

Hyperparameter tuning and performance comparison will be conducted using MLflow experiment tracking.

---

## 📊 Evaluation Metrics

Model performance will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The best-performing model will be selected based on overall generalization performance.

---

## Experiment Tracking

All experiments, parameters, metrics and models will be tracked using MLflow (research branch only).

The final selected model will be registered in the MLflow Model Registry.

---

## Deployment Demonstration

A demonstration video will show:

- Real-time hand landmark extraction using MediaPipe
- Gesture prediction per frame
- Stabilized output using a sliding window mode technique

---

##  Repository Branch Strategy

- `main` branch → Clean ML1 project deliverables (no MLflow code)
- `research` branch → Full MLflow integration and experiment tracking

---

## Status

Project initialization phase — development in progress