from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
FRONTEND_PATH = ROOT / "frontend"

app = FastAPI(title="Heart Disease Prediction System")

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Trained model not found. Run `python train_model.py` before starting the server."
    )

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]

class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: str
    cp: str
    trestbps: Optional[float] = None
    chol: Optional[float] = None
    fbs: Optional[bool] = None
    restecg: Optional[str] = None
    thalch: Optional[float] = None
    exang: Optional[bool] = None
    oldpeak: Optional[float] = None
    slope: Optional[str] = None
    ca: Optional[float] = None
    thal: Optional[str] = None

@app.get("/")
def home():
    return FileResponse(FRONTEND_PATH / "index.html")

app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Heart Disease Prediction API is running"}

@app.post("/api/predict")
def predict(patient: PatientData):
    try:
        values = patient.model_dump()
        row = pd.DataFrame([values], columns=FEATURES)

        # The training script converts these boolean columns to "True"/"False"
        # before fitting the preprocessing pipeline.
        for col in ["fbs", "exang"]:
            row[col] = row[col].map({True: "True", False: "False"})

        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1]) * 100

        return {
            "prediction": prediction,
            "result": (
                "Heart disease pattern detected by the model"
                if prediction == 1
                else "No heart disease pattern detected by the model"
            ),
            "model_probability_percent": round(probability, 1),
            "notice": "This is an educational machine-learning estimate, not a medical diagnosis."
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process the submitted details: {exc}")
