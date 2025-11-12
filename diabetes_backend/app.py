# ==========================================================
# 🚀 Diabetes Prediction Backend (Hybrid Model)
# Works with Lovable Frontend & Trained Models (.pkl)
# ==========================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# ----------------------------------------------------------
# 1️⃣ Initialize FastAPI
# ----------------------------------------------------------
app = FastAPI(
    title="Diabetes Prediction API (Hybrid Ensemble)",
    version="2.0",
    description="Predicts diabetes using SVM, XGBoost, and Neural Network ensemble."
)

# ----------------------------------------------------------
# 2️⃣ Enable CORS (to connect with Lovable frontend)
# ----------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change later to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------
# 3️⃣ Load Models and Scaler
# ----------------------------------------------------------
try:
    svm_model = joblib.load("svm_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    nn_model = joblib.load("nn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Models and Scaler loaded successfully!")
except Exception as e:
    print("❌ Error loading models:", e)


# ----------------------------------------------------------
# 4️⃣ Helper Functions for Encoding
# ----------------------------------------------------------
def encode_gender(gender: str) -> int:
    """Encodes gender string to integer"""
    gender = gender.strip().lower()
    if gender == "male":
        return 1
    elif gender == "female":
        return 0
    else:
        return 2  # For 'Other' or missing

def encode_smoking(smoking_history: str) -> int:
    """Encodes smoking history string to integer"""
    mapping = {
        "never": 0,
        "former": 1,
        "current": 2,
        "ever": 3,
        "not current": 4,
        "no info": 5
    }
    return mapping.get(smoking_history.strip().lower(), 5)


# ----------------------------------------------------------
# 5️⃣ Root Route
# ----------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "🚀 Diabetes Prediction API is running successfully!",
        "info": "Use POST /predict with patient details to get prediction."
    }


# ----------------------------------------------------------
# 6️⃣ Prediction Route
# ----------------------------------------------------------
@app.post("/predict")
def predict_diabetes(data: dict):
    """
    Expected JSON Input:
    {
      "gender": "Female",
      "age": 45,
      "hypertension": 0,
      "heart_disease": 1,
      "smoking_history": "never",
      "bmi": 25.3,
      "HbA1c_level": 6.5,
      "blood_glucose_level": 140
    }
    """

    try:
        # ------------------------------
        # Encode categorical variables
        # ------------------------------
        gender = encode_gender(data["gender"])
        smoking = encode_smoking(data["smoking_history"])

        # ------------------------------
        # Create feature array
        # ------------------------------
        features = np.array([[
            gender,
            data["age"],
            data["hypertension"],
            data["heart_disease"],
            smoking,
            data["bmi"],
            data["HbA1c_level"],
            data["blood_glucose_level"]
        ]])

        # ------------------------------
        # Scale numeric features
        # ------------------------------
        scaled_features = scaler.transform(features)

        # ------------------------------
        # Individual model predictions
        # ------------------------------
        pred_svm = int(svm_model.predict(scaled_features)[0])
        pred_xgb = int(xgb_model.predict(scaled_features)[0])
        pred_nn = int(nn_model.predict(scaled_features)[0])

        # ------------------------------
        # Majority Vote (Hybrid Ensemble)
        # ------------------------------
        preds = [pred_svm, pred_xgb, pred_nn]
        final_pred = 1 if sum(preds) >= 2 else 0
        result = "Diabetic" if final_pred == 1 else "Non-Diabetic"

        # ------------------------------
        # Return as JSON (Frontend Ready)
        # ------------------------------
        return {
            "status": "success",
            "input_data": data,
            "model_predictions": {
                "SVM": pred_svm,
                "XGBoost": pred_xgb,
                "NeuralNetwork": pred_nn
            },
            "final_result": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }


# ----------------------------------------------------------
# 7️⃣ Run Command
# ----------------------------------------------------------
# Run this in terminal:
# uvicorn app:app --reload
# ----------------------------------------------------------