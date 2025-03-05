import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from pipeline import prepare_data, retrain_model

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn and retraining the model",
    version="1.0.0"
)

# Load artifacts at startup
MODEL_PATH = "churn_model.joblib"
ENCODER_PATH = "encoder.joblib"
SCALER_PATH = "scaler.joblib"
FEATURE_NAMES_PATH = "feature_names.joblib"
redundant_features = [
            "Total day charge", "Total eve charge",
            "Total night charge", "Total intl charge"
        ]




try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    logger.info("Artifacts loaded successfully")
    logger.info(f"Expected feature names: {feature_names}")
except Exception as e:
    logger.error(f"Failed to load artifacts: {e}")
    raise



# Pydantic model for prediction input (no Churn, no redundant features)
class PredictInput1(BaseModel):
    State: str
    Account_length: float
    Area_code: float
    International_plan: str
    Voice_mail_plan: str
    Number_vmail_messages: float
    Total_day_minutes: float
    Total_day_calls: float
    Total_eve_minutes: float
    Total_eve_calls: float
    Total_night_minutes: float
    Total_night_calls: float
    Total_intl_minutes: float
    Total_intl_calls: float
    Customer_service_calls: float

    class Config:
        schema_extra = {
            "example": {
                "State": "CA",
                "Account_length": 128.0,
                "Area_code": 415.0,
                "International_plan": "no",
                "Voice_mail_plan": "yes",
                "Number_vmail_messages": 25.0,
                "Total_day_minutes": 265.1,
                "Total_day_calls": 110.0,
                "Total_eve_minutes": 197.4,
                "Total_eve_calls": 99.0,
                "Total_night_minutes": 244.7,
                "Total_night_calls": 91.0,
                "Total_intl_minutes": 10.0,
                "Total_intl_calls": 3.0,
                "Customer_service_calls": 1.0
            }
        }

        
class PredictInput(BaseModel):
    # Mirror raw data schema exactly
    State: str = Field(..., alias="State")
    Account_length: float = Field(..., alias="Account length")
    Area_code: float = Field(..., alias="Area code")
    International_plan: str = Field(..., alias="International plan")
    Voice_mail_plan: str = Field(..., alias="Voice mail plan")
    Number_vmail_messages: float = Field(..., alias="Number vmail messages")
    Total_day_minutes: float = Field(..., alias="Total day minutes")
    Total_day_calls: float = Field(..., alias="Total day calls")
    Total_eve_minutes: float = Field(..., alias="Total eve minutes")
    Total_eve_calls: float = Field(..., alias="Total eve calls")
    Total_night_minutes: float = Field(..., alias="Total night minutes")
    Total_night_calls: float = Field(..., alias="Total night calls")
    Total_intl_minutes: float = Field(..., alias="Total intl minutes")
    Total_intl_calls: float = Field(..., alias="Total intl calls")
    Customer_service_calls: float = Field(..., alias="Customer service calls")

    class Config:
        allow_population_by_field_name = True

def strict_preprocessing(data: dict) -> pd.DataFrame:
    """Mirror pipeline's prepare_data() exactly"""
    # Convert to DataFrame with original column names
    df = pd.DataFrame([data]).rename(columns={
        "International_plan": "International plan",
        "Voice_mail_plan": "Voice mail plan",
        "Account_length": "Account length",
        "Area_code": "Area code",
        "Number_vmail_messages": "Number vmail messages",
        "Total_day_minutes": "Total day minutes",
        "Total_day_calls": "Total day calls",
        "Total_eve_minutes": "Total eve minutes",
        "Total_eve_calls": "Total eve calls",
        "Total_night_minutes": "Total night minutes",
        "Total_night_calls": "Total night calls",
        "Total_intl_minutes": "Total intl minutes",
        "Total_intl_calls": "Total intl calls",
        "Customer_service_calls": "Customer service calls"
    })

    # 1. Encode categoricals
    cat_cols = ["International plan", "Voice mail plan"]
    df[cat_cols] = encoder.transform(df[cat_cols])

    # 2. One-hot encode State
    df = pd.get_dummies(df, columns=["State"], prefix="State")
    
    # 3. Add missing state columns
    expected_states = [col for col in feature_names if col.startswith("State_")]
    for col in expected_states:
        if col not in df.columns:
            df[col] = 0

    # 4. Drop redundant features BEFORE scaling (critical!)
    df = df.drop(columns=redundant_features, errors="ignore")

    # 4.5 Enforce feature order before scaling
    df = df.reindex(columns=feature_names, fill_value=0)

    # 5. Add dummy Churn column for scaling compatibility
    #df["Churn"] = 0  # Scaler expects this column from training

    # 6. Apply scaling
    scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(scaled, columns=df.columns)

    # 7. Remove Churn column post-scaling
    df_scaled = df_scaled.drop(columns=["Churn"], errors="ignore")

    # 8. Enforce feature order and existence
    missing = set(feature_names) - set(df_scaled.columns)
    extra = set(df_scaled.columns) - set(feature_names)
    
    if missing or extra:
        raise ValueError(f"Feature mismatch. Missing: {missing}, Extra: {extra}")

    return df_scaled[feature_names]


# Pydantic model for retrain input
class RetrainInput(BaseModel):
    C: Optional[float] = 1.0
    kernel: Optional[str] = "rbf"
    gamma: Optional[str] = "scale"


# Prediction endpoint
@app.post("/predict")
async def predict(input_data: PredictInput):
    """Main prediction endpoint"""
    logger.info("Starting prediction processing")
        
    # Convert to dict preserving aliases
    raw_data = input_data.model_dump(by_alias=True)
    logger.info(f"Received data: {raw_data}")
        
    try:
        # Process data through pipeline-identical steps
        processed_data = strict_preprocessing(raw_data)
  
        # Make prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[:, 1]
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    

"""
@app.post("/predict", response_model=Dict[str, str])
def predict(data: PredictInput):
    #Predict customer churn based on input features.

    try:
        logger.info("Received prediction request")
        # Preprocess the input
        processed_data = preprocess_input(data)
        logger.info(f"///// /n after processed_data data content: {processed_data}")

        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        result = "Customer Will Churn" if prediction == 1 else "Customer Will Not Churn"

        logger.info(f"Prediction: {result}, Probability: {probability:.2f}")
        return {
            "prediction": result,
            "churn_probability": f"{probability:.2f}"
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
"""

# Retrain endpoint (Excellence)
@app.post("/retrain", response_model=Dict[str, str])
def retrain(data: RetrainInput):
    """Retrain the model with new hyperparameters."""
    try:
        logger.info(f"Retraining model with C={data.C}, kernel={data.kernel}, gamma={data.gamma}")
        retrain_model(C=data.C, kernel=data.kernel, gamma=data.gamma)
        global model
        model = joblib.load(MODEL_PATH)
        logger.info("Model retrained and reloaded successfully")
        return {"status": "Model retrained successfully"}
    except Exception as e:
        logger.error(f"Retrain error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retrain error: {str(e)}")

# Optional endpoint to get feature names
@app.get("/features")
def get_features():
    """Return the list of expected features."""
    return {"features": feature_names}