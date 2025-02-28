from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("churn_model.joblib")
# Load the scaler and feature names
scaler = joblib.load("scaler.joblib")
feature_names = joblib.load("feature_names.joblib")
# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    account_length: float
    area_code: float
    international_plan: float
    voice_mail_plan: float
    number_vmail_messages: float
    total_day_minutes: float
    total_day_calls: float
    total_eve_minutes: float
    total_eve_calls: float
    total_night_minutes: float
    total_night_calls: float
    total_intl_minutes: float
    total_intl_calls: float
    customer_service_calls: float
    state_AK: float
    state_AL: float
    state_AR: float
    state_AZ: float
    state_CA: float
    state_CO: float
    state_CT: float
    state_DC: float
    state_DE: float
    state_FL: float
    state_GA: float
    state_HI: float
    state_IA: float
    state_ID: float
    state_IL: float
    state_IN: float
    state_KS: float
    state_KY: float
    state_LA: float
    state_MA: float
    state_MD: float
    state_ME: float
    state_MI: float
    state_MN: float
    state_MO: float
    state_MS: float
    state_MT: float
    state_NC: float
    state_ND: float
    state_NE: float
    state_NH: float
    state_NJ: float
    state_NM: float
    state_NV: float
    state_NY: float
    state_OH: float
    state_OK: float
    state_OR: float
    state_PA: float
    state_RI: float
    state_SC: float
    state_SD: float
    state_TN: float
    state_TX: float
    state_UT: float
    state_VA: float
    state_VT: float
    state_WA: float
    state_WI: float
    state_WV: float
    state_WY: float


@app.post("/predict")
def predict(data: InputData):
    # Convertir les données en dictionnaire
    input_dict = data.dict()

    # Créer un tableau dans l'ordre des colonnes d'entraînement
    features = [input_dict[col] for col in feature_names]

    # Appliquer le scaling
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Prédiction
    prediction = model.predict(X_scaled)[0]
    result = "Customer Will Churn" if prediction == 1 else "Customer Will Not Churn"
    return {"prediction": result}
