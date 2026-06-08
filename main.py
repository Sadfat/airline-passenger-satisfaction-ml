from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal
import numpy as np
import pandas as pd
import joblib
import os

app = FastAPI(
    title="✈️ Airline Satisfaction Predictor API",
    description="Predict passenger satisfaction probability using a trained XGBoost model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Load model on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

FEATURE_COLS = [
    'Customer Type Enc','Age','Travel Type Enc',
    'Class_Business','Class_Eco','Class_Eco Plus','Flight Distance',
    'Seat comfort','Departure/Arrival time convenient','Food and drink',
    'Gate location','Inflight wifi service','Inflight entertainment',
    'Online support','Ease of Online booking','On-board service',
    'Leg room service','Baggage handling','Checkin service',
    'Cleanliness','Online boarding',
    'Departure Delay in Minutes','Arrival Delay in Minutes'
]

class PassengerInput(BaseModel):
    customer_type: Literal["loyal", "disloyal"] = Field(..., example="loyal")
    age: int = Field(..., ge=5, le=100, example=35)
    travel_type: Literal["business", "personal"] = Field(..., example="business")
    flight_class: Literal["Business", "Eco", "Eco Plus"] = Field(..., example="Business")
    flight_distance: int = Field(..., ge=50, le=10000, example=1500)
    seat_comfort: int = Field(..., ge=0, le=5, example=4)
    timing_convenience: int = Field(..., ge=0, le=5, example=3)
    food_and_drink: int = Field(..., ge=0, le=5, example=3)
    gate_location: int = Field(..., ge=0, le=5, example=3)
    inflight_wifi: int = Field(..., ge=0, le=5, example=3)
    inflight_entertainment: int = Field(..., ge=0, le=5, example=5)
    online_support: int = Field(..., ge=0, le=5, example=4)
    ease_of_booking: int = Field(..., ge=0, le=5, example=4)
    onboard_service: int = Field(..., ge=0, le=5, example=4)
    leg_room: int = Field(..., ge=0, le=5, example=4)
    baggage_handling: int = Field(..., ge=0, le=5, example=4)
    checkin_service: int = Field(..., ge=0, le=5, example=4)
    cleanliness: int = Field(..., ge=0, le=5, example=4)
    online_boarding: int = Field(..., ge=0, le=5, example=4)
    departure_delay: int = Field(0, ge=0, le=1440, example=0)
    arrival_delay: int = Field(0, ge=0, le=1440, example=0)

class PredictionResponse(BaseModel):
    satisfaction_probability: float
    predicted_class: str
    confidence: str
    risk_tier: str
    recommendation: str

def build_feature_vector(p: PassengerInput) -> pd.DataFrame:
    row = {
        'Customer Type Enc': 1 if p.customer_type == "loyal" else 0,
        'Age': p.age,
        'Travel Type Enc': 1 if p.travel_type == "business" else 0,
        'Class_Business': 1 if p.flight_class == "Business" else 0,
        'Class_Eco': 1 if p.flight_class == "Eco" else 0,
        'Class_Eco Plus': 1 if p.flight_class == "Eco Plus" else 0,
        'Flight Distance': p.flight_distance,
        'Seat comfort': p.seat_comfort,
        'Departure/Arrival time convenient': p.timing_convenience,
        'Food and drink': p.food_and_drink,
        'Gate location': p.gate_location,
        'Inflight wifi service': p.inflight_wifi,
        'Inflight entertainment': p.inflight_entertainment,
        'Online support': p.online_support,
        'Ease of Online booking': p.ease_of_booking,
        'On-board service': p.onboard_service,
        'Leg room service': p.leg_room,
        'Baggage handling': p.baggage_handling,
        'Checkin service': p.checkin_service,
        'Cleanliness': p.cleanliness,
        'Online boarding': p.online_boarding,
        'Departure Delay in Minutes': p.departure_delay,
        'Arrival Delay in Minutes': p.arrival_delay,
    }
    return pd.DataFrame([row])[FEATURE_COLS]

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model_loaded": model is not None,
            "docs": "/docs", "version": "1.0.0"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(passenger: PassengerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run /train first.")
    X = build_feature_vector(passenger)
    prob = float(model.predict_proba(X)[0, 1])
    pred = "satisfied" if prob >= 0.5 else "dissatisfied"
    if prob >= 0.85:
        confidence, risk_tier = "Very High", "Low Risk"
        rec = "Passenger is very likely satisfied. Include in NPS champion outreach."
    elif prob >= 0.65:
        confidence, risk_tier = "High", "Low Risk"
        rec = "Likely satisfied. Minor service tweaks could convert to promoter."
    elif prob >= 0.45:
        confidence, risk_tier = "Moderate", "Medium Risk"
        rec = "Borderline case. Prioritise entertainment and seat comfort improvements."
    elif prob >= 0.25:
        confidence, risk_tier = "High", "High Risk"
        rec = "Likely dissatisfied. Trigger post-flight recovery offer within 24h."
    else:
        confidence, risk_tier = "Very High", "Critical Risk"
        rec = "Very high dissatisfaction risk. Escalate to customer relations immediately."
    return PredictionResponse(
        satisfaction_probability=round(prob, 4),
        predicted_class=pred,
        confidence=confidence,
        risk_tier=risk_tier,
        recommendation=rec,
    )

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(passengers: list[PassengerInput]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(passengers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit is 1000.")
    rows = pd.concat([build_feature_vector(p) for p in passengers], ignore_index=True)
    probs = model.predict_proba(rows)[:, 1].tolist()
    return [{"satisfaction_probability": round(p, 4),
             "predicted_class": "satisfied" if p >= 0.5 else "dissatisfied"} for p in probs]

@app.get("/model/info", tags=["Model"])
def model_info():
    return {
        "model_type": "XGBoost Classifier",
        "n_features": len(FEATURE_COLS),
        "features": FEATURE_COLS,
        "performance": {"accuracy": 0.9619, "precision": 0.9617, "recall": 0.9609, "roc_auc": 0.9940, "f1": 0.9611},
        "trained_on": "129,880 airline passengers",
        "threshold": 0.5,
    }
