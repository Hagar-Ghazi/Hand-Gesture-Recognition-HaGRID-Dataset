# Imports

import os
import joblib
import mlflow
import mlflow.data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from MLflow_utils import setup_mlflow_experiment



# --- MLflow Configuration ---
mlflow.set_tracking_uri("file:./mlruns")
EXPERIMENT_NAME = "Hand_Gesture_Recognition_Model_Comparison_Study"
setup_mlflow_experiment(EXPERIMENT_NAME)



# --- Dataset Loading ---
DATA_PATH  = r"C:\Users\WellCome\Downloads\Hand Gesture Detection and Classification Project\Data\hand_landmarks_data.csv"
SPLIT_PATH = r"C:\Users\WellCome\Downloads\Hand Gesture Detection and Classification Project\Data\dataset_splits.pkl"

df = pd.read_csv(DATA_PATH)
data = joblib.load(SPLIT_PATH)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]



# High-level tracking for MLflow UI
mlflow_dataset = mlflow.data.from_pandas(df, source=DATA_PATH, name="Hand_Gesture_CSV")



# --- Save Dataset Info ---
DATA_INFO_PATH = "Data_Info.txt"
with open(DATA_INFO_PATH, "w") as f:
    f.write("Hand Gesture Recognition Dataset Info\n" + "="*50 + "\n")
    f.write(f"Dataset Shape: {df.shape}\n")
    # FIX for image_7226ff.png: Use np.unique for numpy arrays
    f.write(f"Classes: {np.unique(y_test).tolist()}\n")



# --- SVM Hyperparameter Tuning ---
param_grid = {'kernel': ['rbf'], 'C': [100], 'gamma': [0.1]}
base_svc = SVC(class_weight='balanced')
svm_gs = GridSearchCV(estimator=base_svc, param_grid=param_grid, cv=3)
print("Running GridSearch for SVM...")
svm_gs.fit(X_train, y_train)
best_svm = svm_gs.best_estimator_



# --- Models for Comparison ---
models_to_run = [
    ("SVM_Tuned", best_svm, svm_gs.best_params_),
    ("Random_Forest", RandomForestClassifier(n_estimators=100), {"n_estimators": 100}),
    ("Logistic_Regression", LogisticRegression(max_iter=1000), {"max_iter": 1000}),
    ("XGBoost", XGBClassifier(eval_metric="mlogloss"), {"eval_metric": "mlogloss"})

]



# --- Training & Logging ---
results = []
for name, model, params in models_to_run:
    print(f"\nRunning: {name}")
    with mlflow.start_run(run_name=name):
        # Log ACTUAL DATA as artifact and input
        mlflow.log_artifact(DATA_PATH)
        mlflow.log_input(mlflow_dataset, context="training")
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Calculate All Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average='weighted'),
            "recall": recall_score(y_test, preds, average='weighted'),
            "f1_score": f1_score(y_test, preds, average='weighted')
        }
        
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {name}")
        cm_path = f"cm_{name}.png"
        plt.savefig(cm_path)
        plt.close()
        
        mlflow.log_artifact(cm_path)
        mlflow.sklearn.log_model(model, "model")
        results.append((name, metrics["accuracy"], model))

# --- Winner Selection ---
best_name, _, final_model = max(results, key=lambda x: x[1])
joblib.dump(final_model, "best_model.pkl")
print(f"\nWINNER: {best_name} (Saved as best_model.pkl)")