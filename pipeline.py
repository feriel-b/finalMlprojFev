import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import mlflow
import mlflow.sklearn
import time

# warnings.filterwarnings("ignore")

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Churn_Pred")


def prepare_data(train_path="churn_80.csv", test_path="churn_20.csv"):
    """Loads, cleans, and prepares data for training and evaluation using original ordinal encoding for categorical features."""
    df_80 = pd.read_csv(train_path)
    df_20 = pd.read_csv(test_path)

    # Fill missing values with mean for numeric columns
    for col in df_80.select_dtypes(include=["float64", "int64"]).columns:
        df_80[col].fillna(df_80[col].mean(), inplace=True)
        df_20[col].fillna(df_20[col].mean(), inplace=True)

    # Identify categorical features (including 'State')
    categorical_features = ["International plan", "Voice mail plan"]

    # Initialize the OrdinalEncoder and apply it to both datasets
    encoder = OrdinalEncoder()
    df_80[categorical_features] = encoder.fit_transform(df_80[categorical_features])
    df_20[categorical_features] = encoder.transform(df_20[categorical_features])

    # One-hot encode the "State" feature (to generate multiple columns, e.g., state_0 to state_6).
    df_80 = pd.get_dummies(df_80, columns=["State"], prefix="state")
    df_20 = pd.get_dummies(df_20, columns=["State"], prefix="state")

    # Convert the Churn feature to int (if necessary)
    df_80["Churn"] = df_80["Churn"].astype(int)
    df_20["Churn"] = df_20["Churn"].astype(int)

    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler()
    df_80_scaled = pd.DataFrame(scaler.fit_transform(df_80), columns=df_80.columns)
    df_20_scaled = pd.DataFrame(scaler.transform(df_20), columns=df_20.columns)

    # Drop redundant features if needed
    drop_cols = [
        "Total day charge",
        "Total eve charge",
        "Total night charge",
        "Total intl charge",
    ]
    df_80_scaled.drop(columns=drop_cols, inplace=True, errors="ignore")
    df_20_scaled.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Separate features and labels
    X_train = df_80_scaled.drop(columns=["Churn"])
    y_train = df_80_scaled["Churn"]
    X_test = df_20_scaled.drop(columns=["Churn"])
    y_test = df_20_scaled["Churn"]

    pd.set_option("display.max_columns", None)

    print(" Data preparation !")
    print(X_train.head())
    # feature names
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, "feature_names.joblib")

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, X_test, y_test, C=1.0, kernel="rbf", gamma="scale"):
    """Trains an SVM model and logs with MLflow, returns (model, test_accuracy)."""

    with mlflow.start_run():
        model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42, probability=True)
        model.fit(X_train, y_train)

        # Log hyperparameters for SVM
        mlflow.log_param("C", C)
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("gamma", gamma)

        # Log metrics
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        if "test_accuracy" not in mlflow.active_run().data.metrics:
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)

        # Log the model with MLflow
        mlflow.sklearn.log_model(model, "svm_model")

        # Save locally with a filename based on hyperparameters
        joblib.dump(model, f"churn_model_{C}_{kernel}_{gamma}.joblib")
        print(
            f"✅ Model trained and logged with MLflow (C={C}, kernel={kernel}, gamma={gamma})"
        )
        #  Register and Promote the Model 
        client = MlflowClient()
        model_name = "ChurnPredictionSVM"
        # Create the registered model if it doesn't exist
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)
        
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/svm_model"
        # Register the model
        registration_result = mlflow.register_model(model_uri, model_name)
        # Optionally wait for the registration to complete
        time.sleep(5)
        try:
            client.transition_model_version_stage(
                name=model_name, version=registration_result.version, stage="Production"
            )
            print(f"✅ Model version {registration_result.version} promoted to Production")
        except Exception as e:
            print("Model promotion skipped:", e)

    return model, test_acc


def save_model(model, filename="churn_model.joblib"):
    """Saves the given model to a file."""
    joblib.dump(model, filename)
    print(f"💾 Model saved as {filename}")


def load_model(model_path="churn_model.joblib"):
    """Loads the trained model."""
    try:
        return mlflow.sklearn.load_model("models:/ChurnPredictionSVM/Production")
    except Exception as e:
        raise ValueError(f"Erreur de chargement : {str(e)}")


def retrain_model(C=1.0, kernel="rbf", gamma="scale"):
    """Retrains the SVM model with new hyperparameters."""
    X_train, y_train, _, _ = prepare_data()
    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "churn_model.joblib")
    print("✅ Model retrained and saved!")


def evaluate_model(model, X_test, y_test):
    """Evaluates the model on test data and prints metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.2f}")
    print("\n🔍 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    return acc, y_pred


def plot_confusion_matrix(
    y_true, y_pred, classes=["No Churn", "Churn"], filename="confusion_matrix.png"
):
    """
    Plots and saves a confusion matrix as an image file,
    and logs it as an artifact to MLflow.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"💾 Confusion matrix saved as {filename}")
    plt.close()
    mlflow.log_artifact(
        filename
    )  # Log the confusion matrix image as an artifact in MLflow


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(y_true, y_score, filename="roc_curve.png"):
    """
    Plots the ROC curve and saves it as an image file,
    and logs it as an artifact in MLflow.

    Parameters:
    - y_true: Array-like of true binary labels.
    - y_score: Array-like of predicted probabilities for the positive class.
    - filename: Filename for the saved ROC curve image.
    """
    # Compute ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"💾 ROC curve saved as {filename}")
    plt.close()

    # Log the ROC curve image as an artifact in MLflow
    mlflow.log_artifact(filename)


# """Retrains the SVM model with new hyperparameters and saves it as the default model."""
"""
def retrain_model(C=1.0, kernel='rbf', gamma='scale'):
    X_train, y_train, _, _ = prepare_data()
    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "churn_model.joblib")
    print("✅ Model retrained and saved!")
"""
