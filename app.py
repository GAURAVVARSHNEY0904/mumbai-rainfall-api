from fastapi import FastAPI
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load trained artifacts
model = joblib.load("rainfall_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("month_encoder.pkl")

@app.get("/")
def home():
    return {"status": "Mumbai Rainfall Prediction API is running"}

@app.post("/predict")
def predict(year: int, month: str):
    month = month.lower()
    month_encoded = encoder.transform([month])[0]

    input_data = scaler.transform([[year, month_encoded]])
    prediction = model.predict(input_data)

    return {
        "year": year,
        "month": month,
        "predicted_rainfall_mm": round(float(prediction[0]), 2)
    }