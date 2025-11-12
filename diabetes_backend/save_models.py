import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler # Added RobustScaler for consistency
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from scipy.stats import mode
import matplotlib.pyplot as plt # Kept for completeness, though not strictly needed for saving

# --- 1. DEFINE YOUR MODEL COMPONENTS AND PIPELINE ---

# Define columns (Based on previous mock data structure)
numerical_features = ['age', 'annual_income']
categorical_features = ['city', 'employment_type']
target = 'is_approved'

# Preprocessing components (using RobustScaler as suggested by your imports)
scaler = RobustScaler()
encoder = OneHotEncoder(handle_unknown='ignore')

# Preprocessor that combines the steps (MUST be saved for the API)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_features),
        ('cat', encoder, categorical_features)
    ],
    remainder='passthrough'
)

# Define the HybridEnsembleModel class (Must be defined here and in app.py)
class HybridEnsembleModel:
    """Custom class to handle hard voting for models, including SVC without predict_proba."""
    def __init__(self, svm, nn, xgb):
        self.svm = svm
        self.nn = nn
        self.xgb = xgb

    def predict(self, X):
        svm_pred = self.svm.predict(X)
        nn_pred = self.nn.predict(X)
        xgb_pred = self.xgb.predict(X)

        # Stack predictions and find the mode (majority vote)
        stacked = np.vstack([svm_pred, nn_pred, xgb_pred])
        majority_vote, _ = mode(stacked, axis=0)
        return majority_vote.flatten()


# --- 2. DATA LOADING AND TRAINING LOGIC (Adapted from your ensemble code) ---

print("Generating mock data, fitting preprocessor, and training models...")

# Mock Data Generation (To make the script runnable)
data = {
    'age': np.random.randint(20, 60, 500),
    'annual_income': np.random.randint(30000, 150000, 500),
    'city': np.random.choice(['New York', 'London', 'Tokyo', 'Paris'], 500),
    'employment_type': np.random.choice(['Full-Time', 'Part-Time', 'Self-Employed'], 500),
    'is_approved': np.random.choice([0, 1], p=[0.8, 0.2], size=500) # Simulating imbalance
}
df = pd.DataFrame(data)

X = df.drop(target, axis=1)
y = df[target]

# Split data before applying preprocessor for a fair test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Fit Preprocessor on Training Data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Convert back to DataFrame (optional, but helpful if we need feature names later)
# We need to use NumPy arrays for SMOTETomek and model training
X_res, y_res = X_train_transformed, y_train # Simplified, skipping SMOTETomek for saving

# We use the full transformed training set for all models now
X_res_scaled = X_res
X_svm_train = X_res
y_svm_train = y_res
X_test_scaled = X_test_transformed


# Define Individual Models
models = {
    "SVM": SVC(probability=False, kernel='rbf', random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

# Train Individual Models
print("\n=== Training SVM ===")
models["SVM"].fit(X_svm_train, y_svm_train)
print("SVM training completed.")

print("\n=== Training Neural Network ===")
models["Neural Network"].fit(X_res_scaled, y_res)
print("Neural Network training completed.")

print("\n=== Training XGBoost ===")
models["XGBoost"].fit(X_res_scaled, y_res)
print("XGBoost training completed.")

# --- 3. SAVE THE FINAL COMPONENTS ---
print("\n--- Starting Final File Save ---")

# 1. Save the ColumnTransformer (Pre-processor)
joblib.dump(preprocessor, "preprocessor.pkl")
print("Saved: preprocessor.pkl (Contains Scaler and OneHotEncoder)")

# 2. Save the SVM model
joblib.dump(models['SVM'], "svm_model.pkl")
print("Saved: svm_model.pkl")

# 3. Save the XGBoost model
joblib.dump(models['XGBoost'], "xgb_model.pkl")
print("Saved: xgb_model.pkl")

# 4. Save the Neural Network (MLPClassifier) model
joblib.dump(models['Neural Network'], "nn_model.pkl")
print("Saved: nn_model.pkl")

print("\n🎉 Model Saving Complete!")
print("Four files created: preprocessor.pkl, svm_model.pkl, xgb_model.pkl, nn_model.pkl")