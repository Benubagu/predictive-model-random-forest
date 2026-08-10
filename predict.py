"""
predict.py
-----------
Command-line interface for diagnosing a new patient using the
trained Random Forest model.

Usage:
    python3 predict.py
    (then answer the prompts for each clinical attribute)

Or import predict_patient() into other code / a web interface.
"""

import joblib
import pandas as pd
from preprocessing import FEATURE_NAMES, FEATURE_DESCRIPTIONS

MODEL_PATH = "model/random_forest_heart_model.joblib"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_patient(model, patient: dict):
    """
    patient: dict with keys matching FEATURE_NAMES
    Returns: (prediction_label, probability_of_disease)
    """
    X = pd.DataFrame([patient])[FEATURE_NAMES]
    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= 0.5)
    label = "Heart Disease Likely" if pred == 1 else "No Heart Disease Indicated"
    return label, proba


def prompt_for_patient():
    print("Enter patient clinical data:\n")
    patient = {}
    for feat in FEATURE_NAMES:
        desc = FEATURE_DESCRIPTIONS[feat]
        while True:
            raw = input(f"{feat} ({desc}): ").strip()
            try:
                patient[feat] = float(raw)
                break
            except ValueError:
                print("  Please enter a numeric value.")
    return patient


if __name__ == "__main__":
    model = load_model()
    patient = prompt_for_patient()
    label, proba = predict_patient(model, patient)
    print(f"\nPrediction: {label}")
    print(f"Predicted probability of heart disease: {proba:.1%}")
