import mlflow
import mlflow.sklearn
import mlflow.data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)



def setup_mlflow_experiment(experiment_name):
    mlflow.set_experiment(experiment_name)

def log_model_run(run_name, model, params,
                  X_train, y_train, X_test, y_test,
                  dataset_df, dataset_info_path):

    with mlflow.start_run(run_name=run_name):

        # Log dataset
        dataset = mlflow.data.from_pandas(dataset_df, name="Hand_Gesture_Dataset")
        mlflow.log_input(dataset, context="training")

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Metrics
        metrics = {
            "training_accuracy": accuracy_score(y_train, train_preds),
            "test_accuracy": accuracy_score(y_test, test_preds),
            "precision": precision_score(y_test, test_preds, average='weighted'),
            "recall": recall_score(y_test, test_preds, average='weighted'),
            "f1_score": f1_score(y_test, test_preds, average='weighted')
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "gesture_model")
        mlflow.log_artifact(dataset_info_path)

        # Classification report
        report = classification_report(y_test, test_preds)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # Confusion Matrix
        cm = confusion_matrix(y_test, test_preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {run_name}")
        plt.tight_layout()

        cm_filename = "confusion_matrix.png"
        plt.savefig(cm_filename)
        plt.close()

        mlflow.log_artifact(cm_filename)

        print(f"Run '{run_name}' successfully logged to MLflow.")