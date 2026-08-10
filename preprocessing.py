"""
preprocessing.py
-----------------
Loads and cleans the heart disease dataset for the
Random Forest heart disease diagnosis system.

Dataset: Cleveland Heart Disease dataset (UCI Machine Learning Repository)
303 patient records, 13 clinical attributes + target label.
"""

import pandas as pd
import numpy as np

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1-4)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting electrocardiographic results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (1-3)",
    "ca": "Number of major vessels colored by fluoroscopy (0-3)",
    "thal": "Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)",
}


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and standardize column names."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
      - Replace '?' missing-value markers with NaN
      - Cast all feature columns to numeric
      - Impute missing values with column median
      - Collapse the multi-class diagnosis (0-4) into a binary
        target: 0 = no heart disease, 1 = heart disease present
    """
    df = df.copy()
    df.replace("?", np.nan, inplace=True)

    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Median imputation for the small number of missing values
    for col in FEATURE_NAMES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Binary target: any diagnosis > 0 means disease present
    df["target"] = (df["diagnosis"] > 0).astype(int)

    return df


def get_features_and_target(df: pd.DataFrame):
    X = df[FEATURE_NAMES]
    y = df["target"]
    return X, y


if __name__ == "__main__":
    raw = load_raw_data("data/heart.csv")
    clean = clean_data(raw)
    print(clean[FEATURE_NAMES + ["target"]].describe())
    print("\nClass balance:")
    print(clean["target"].value_counts())
