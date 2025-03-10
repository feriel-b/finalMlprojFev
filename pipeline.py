import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import time
import os 
import logging
from elasticsearch import Elasticsearch
from datetime import datetime

# --- Elasticsearch and Logging Setup ---
# Initialize Elasticsearch client
elasticsearch_available = False
es = None
try:
    es = Elasticsearch(['http://localhost:9200'], compatibility_mode=True)
    elasticsearch_available = True
except Exception as e:
    print(f"Elasticsearch not available: {e}")
    elasticsearch_available = False

# warnings.filterwarnings("ignore")

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Churn_Pred")


# Custom handler to send logs to Elasticsearch
class ElasticsearchHandler(logging.Handler):
    def __init__(self, es_client, index):
        logging.Handler.__init__(self)
        self.es = es_client
        self.index = index

    def emit(self, record):
        try:

            log_entry = self.format(record)
            timestamp = datetime.fromtimestamp(record.created).isoformat()
            if self.es.ping():  # Check if ES is available
                self.es.index(index=self.index, body={
                    "timestamp": timestamp,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": log_entry
                })
        except Exception as e :
            print(f"Failed to log to Elasticsearch: {e}")
        with open('fallback_logs.log', 'a') as f:
            f.write(f"{timestamp} - {record.levelname} - {log_entry}\n")


        # Configure MLflow logger
mlflow_logger = logging.getLogger("mlflow")
mlflow_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


# Create the index if it doesn't exist
if elasticsearch_available and es is not None:
    try:
        # Create the index if it doesn't exist
        if not es.indices.exists(index='mlflow-metrics'):
            mapping = {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "level": {"type": "keyword"},
                        "logger": {"type": "keyword"},
                        "message": {"type": "text"}
                    }
                }
            }
            es.indices.create(index='mlflow-metrics', body=mapping)
            # Add ES handler
            es_handler = ElasticsearchHandler(es, 'mlflow-metrics')
            es_handler.setFormatter(formatter)
            mlflow_logger.addHandler(es_handler)
    except Exception as e:
        print(f"Failed to set up Elasticsearch logging: {e}")
else:
    # Fallback to console logging if ES is not available
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    mlflow_logger.addHandler(console_handler)
    print("Using console logging as fallback (Elasticsearch not available)")
        




class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Unified preprocessing pipeline for training and inference"""
    def __init__(self):
        self.redundant_features = [
            "Total day charge", "Total eve charge",
            "Total night charge", "Total intl charge"
        ]
        self.categorical_features = ["International plan", "Voice mail plan"]
        self.state_column = "State"
        self.target = "Churn"
        
    def fit(self, X, y=None):
        # Encoders
        self.encoder = OrdinalEncoder()
        self.encoder.fit(X[self.categorical_features])
        
        # Scaler (fitted later in prepare_data)
        self.scaler = MinMaxScaler()
        
        # Save feature names structure
        X_processed = self.transform(X, training=True)
        self.feature_names = X_processed.columns.tolist()
        return self

    def transform(self, X, training=False):
        # Copy and clean
        df = X.copy()
        
        # Fill missing values
        num_cols = df.select_dtypes(include=["number"]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        
        # Encode categoricals
        df[self.categorical_features] = self.encoder.transform(
            df[self.categorical_features]
        )
        
        # One-hot encode state
        df = pd.get_dummies(df, columns=[self.state_column], prefix="State")
        
        # Drop redundant features
        df = df.drop(columns=self.redundant_features, errors="ignore")
        
        # Training-specific operations
        if training:
            # Fit scaler on cleaned data
            self.scaler.fit(df)

             # Assign feature names from the current dataframe
            self.feature_names = df.columns.tolist()
    
            
            # Save expected columns
            joblib.dump(self.feature_names, "feature_names.joblib")
            joblib.dump(self.redundant_features, "redundant_features.joblib")
            joblib.dump(self.encoder, "encoder.joblib")
            joblib.dump(self.scaler, "scaler.joblib")
            
        # Scale and return
        scaled = self.scaler.transform(df)
        return pd.DataFrame(scaled, columns=df.columns)
    
  

def prepare_data(train_path="churn_80.csv", test_path="churn_20.csv"):
    """Standardized data preparation"""
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Initialize preprocessor
    preprocessor = FeaturePreprocessor()
    
    # Fit and transform
    X_train = preprocessor.fit_transform(train.drop(columns=["Churn"]))
    X_test = preprocessor.transform(test.drop(columns=["Churn"]))
    
    # Get targets
    y_train = train["Churn"].values
    y_test = test["Churn"].values
    
    return X_train, y_train, X_test, y_test



def train_model(X_train, y_train, X_test, y_test, C=1.0, kernel="rbf", gamma="scale"):
    """Trains an SVM model and logs with MLflow, returns (model, test_accuracy)."""
    with mlflow.start_run() as run:  # Start a new MLflow run
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
        # Wait for the registration to complete
        time.sleep(5)
        try:
            client.transition_model_version_stage(
                name=model_name, version=registration_result.version, stage="Production"
            )
            mlflow_logger.info(f"Model version {registration_result.version} promoted to Production")
            print(f"✅ Model version {registration_result.version} promoted to Production")
        except Exception as e:
            print("Model promotion skipped:", e)
    return model, test_acc



def save_model(model, filename="churn_model.joblib"):
    """Saves the given model to a file."""
    joblib.dump(model, filename)
    mlflow_logger.info(f"Model saved as {filename}")
    print(f"💾 Model saved as {filename}")


def load_model(model_path="churn_model.joblib"):
    """Loads the trained model."""
    try:
        model = mlflow.sklearn.load_model("models:/ChurnPredictionSVM/Production")
        mlflow_logger.info("Model loaded from Production stage")
        return model
    except Exception as e:
        mlflow_logger.error(f"Model loading failed: {str(e)}")
        raise ValueError(f"Erreur de chargement : {str(e)}")




def evaluate_model(model, X_test, y_test):
    """Evaluates the model on test data and prints metrics."""
   

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow_logger.info(f"Model evaluation - Accuracy: {acc:.2f}")
    print(f"✅ Accuracy: {acc:.2f}")
    print("\n🔍 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], zero_division=0))
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
    mlflow_logger.info(f"Confusion matrix saved as {filename}")
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

def retrain_model(C=1.0, kernel='rbf', gamma='scale'):
    X_train, y_train, _, _ = prepare_data()
    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42, probability=True)
    model.fit(X_train, y_train)
    joblib.dump(model, "churn_model.joblib")
    mlflow_logger.info(f"Model retrained with C={C}, kernel={kernel}, gamma={gamma}")
    print("✅ Model retrained and saved!")

