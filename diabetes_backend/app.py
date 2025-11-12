from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
# REMOVED: from scipy.stats import mode 

# -----------------------------------------
# ✅ IMPORT the class from its own module
# -----------------------------------------
from models.model_definitions import HybridEnsembleModel # <-- Make sure this path is correct!

# -----------------------------------------
# Load model + scaler
# -----------------------------------------
model = joblib.load("models/hybrid_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------------------
# FastAPI App Setup (The rest of your code remains the same)
# -----------------------------------------
app = FastAPI()
app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"],

)



class InputData(BaseModel):

    gender: str

    age: float

    hypertension: int

    heart_disease: int

    smoking_history: str

    bmi: float

    HbA1c_level: float

    blood_glucose_level: float



@app.post("/predict")

def predict(data: InputData):

    gender_map = {"Male": 1, "Female": 0, "Other": 2}

    smoking_map = {"never": 0, "current": 1, "former": 2, "ever": 3, "not current": 4}



    X = np.array([[

        gender_map.get(data.gender, 2),

        data.age,

        data.hypertension,

        data.heart_disease,

        smoking_map.get(data.smoking_history, 0),

        data.bmi,

        data.HbA1c_level,

        data.blood_glucose_level,

    ]])



    X = scaler.transform(X)

    prediction = model.predict(X)[0]



    return {"prediction": "Diabetic" if prediction == 1 else "Non-Diabetic"}