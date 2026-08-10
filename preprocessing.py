"""
preprocessing.py
-----------------
Loads and cleans the heart disease dataset for the
Random Forest heart disease diagnosis system.

Dataset: Cleveland Heart Disease dataset (UCI Machine Learning Repository)
303 patient records, 13 clinical attributes + target label.

Note: this module does NOT impute missing values. Imputation is a fold-
dependent statistic (the median must come from the training fold only),
so it lives inside the sklearn Pipeline built in evaluation.py, fit
fresh on each cross-validation split rather than once on the full
dataset before splitting.
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

EXPECTED_ROWS = 303
EXPECTED_COLUMNS = FEATURE_NAMES + ["diagnosis"]

# Known valid values for categorical/discrete columns. Anything outside
# these sets (excluding missing values, which are handled separately)
# means the CSV is not the Cleveland dataset we expect.
VALID_VALUES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
    "diagnosis": {0, 1, 2, 3, 4},
}


def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Fail loudly if the loaded CSV isn't shaped like the Cleveland heart
    disease dataset, rather than silently degrading the model on a
    corrupted or wrong file.
    """
    if df.shape[0] != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, got {df.shape[0]}. "
            "Is this the Cleveland heart.csv?"
        )

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")

    for col, allowed in VALID_VALUES.items():
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        bad = values[~values.isin(allowed)]
        if not bad.empty:
            raise ValueError(
                f"Column '{col}' has values outside {sorted(allowed)}: "
                f"{sorted(bad.unique())}"
            )


def load_raw_data(path: str, validate: bool = True) -> pd.DataFrame:
    """Load the raw CSV, standardize column names, and (by default) run
    validate_raw_data() so a corrupted or wrong file fails immediately."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if validate:
        validate_raw_data(df)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
      - Replace '?' missing-value markers with NaN
      - Cast all feature columns to numeric
      - Collapse the multi-class diagnosis (0-4) into a binary
        target: 0 = no heart disease, 1 = heart disease present

    Missing values are left as NaN — see module docstring.
    """
    df = df.copy()
    df.replace("?", np.nan, inplace=True)

    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary target: any diagnosis > 0 means disease present
    df["target"] = (df["diagnosis"] > 0).astype(int)

    return df


def get_features_and_target(df: pd.DataFrame):
    X = df[FEATURE_NAMES]
    y = df["target"]
    return X, y


if __name__ == "__main__":
    from config import DATA_PATH

    raw = load_raw_data(DATA_PATH)
    clean = clean_data(raw)
    print(clean[FEATURE_NAMES + ["target"]].describe())
    print("\nMissing values per column:")
    print(clean[FEATURE_NAMES].isna().sum())
    print("\nClass balance:")
    print(clean["target"].value_counts())
