import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse

# 1. Initialize FastAPI app
app = FastAPI(
    title="UCI Diabetes Readmission API",
    description="Predicts 30-day hospital readmission risk for diabetic patients.",
    version="1.0.0"
)

# 2. Load the trained pipeline ONCE when the server starts
MODEL_PATH = "models/model.joblib"
model = joblib.load(MODEL_PATH)

# 3. Define the input schema with Pydantic (validates types & ranges)
class PatientData(BaseModel):
    race: str = Field(default="Caucasian", examples=["Caucasian"])
    gender: str = Field(default="Female", examples=["Female"])
    age: str = Field(default="[70-80)", examples=["[70-80)"])
    time_in_hospital: int = Field(default=4, ge=1, le=14)
    num_lab_procedures: int = Field(default=45, ge=0)
    num_procedures: int = Field(default=1, ge=0)
    num_medications: int = Field(default=15, ge=0)
    number_outpatient: int = Field(default=0, ge=0)
    number_emergency: int = Field(default=0, ge=0)
    number_inpatient: int = Field(default=1, ge=0)
    number_diagnoses: int = Field(default=7, ge=1)
    max_glu_serum: str = Field(default="Not_Measured", examples=["Not_Measured"])
    A1Cresult: str = Field(default=">8", examples=[">8"])
    insulin: str = Field(default="Up", examples=["Up"])
    change: str = Field(default="Ch", examples=["Ch"])
    diabetesMed: str = Field(default="Yes", examples=["Yes"])


@app.get("/")
def serve_frontend():
    return FileResponse("app/static/index.html")

# 4. Health check endpoint (used by Docker / cloud platforms to verify the API is alive)
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# 5. Prediction endpoint
@app.post("/predict")
def predict_readmission(patient: PatientData):
    # Convert incoming JSON to a dictionary
    input_data = patient.model_dump()

    # Align with all 43 columns the model expects (missing ones default to NaN for the pipeline's imputer)
    expected_cols = model.feature_names_in_
    row = {col: input_data.get(col, np.nan) for col in expected_cols}
    df = pd.DataFrame([row])

    # Get probability of class 1 (<30 days readmission)
    prob = float(model.predict_proba(df)[0, 1])

    # Assign a human-readable clinical risk tier
    if prob >= 0.60:
        risk_tier = "High"
    elif prob >= 0.35:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    return {
        "readmission_probability": round(prob, 4),
        "risk_tier": risk_tier,
        "prediction": int(prob >= 0.5)
    }