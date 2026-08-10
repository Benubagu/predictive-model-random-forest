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
import numpy as np
import pandas as pd

from config import MODEL_PATH
from preprocessing import FEATURE_NAMES, FEATURE_DESCRIPTIONS


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
    """Prompt for each feature; a blank answer is recorded as missing (NaN)
    and imputed by the saved pipeline, same as during training."""
    print("Enter patient clinical data (leave blank if unknown):\n")
    patient = {}
    for feat in FEATURE_NAMES:
        desc = FEATURE_DESCRIPTIONS[feat]
        while True:
            raw = input(f"{feat} ({desc}): ").strip()
            if not raw:
                patient[feat] = np.nan
                break
            try:
                patient[feat] = float(raw)
                break
            except ValueError:
                print("  Please enter a numeric value, or leave blank if unknown.")
    return patient


if __name__ == "__main__":
    model = load_model()
    patient = prompt_for_patient()
    label, proba = predict_patient(model, patient)
    print(f"\nPrediction: {label}")
    print(f"Predicted probability of heart disease: {proba:.1%}")
