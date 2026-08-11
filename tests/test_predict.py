from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.evaluation import build_pipeline
from src.inference.predict import predict_patient
from src.core.preprocessing import FEATURE_NAMES


@pytest.fixture(scope="module")
def quick_model(clean_df):
    """A pipeline fit directly (no grid search) for fast, deterministic
    regression testing of predict_patient() -- not representative of
    production quality; see evaluation.nested_cv_evaluate for that."""
    X = clean_df[FEATURE_NAMES]
    y = clean_df["target"]
    pipe = build_pipeline(random_state=42)
    pipe.fit(X, y)
    return pipe


def test_predict_patient_ranks_known_patients_correctly(clean_df, quick_model):
    disease_row = clean_df[clean_df["diagnosis"] == 4].iloc[0]  # most severe
    healthy_row = clean_df[clean_df["diagnosis"] == 0].iloc[0]  # confirmed no disease

    disease_patient = disease_row[FEATURE_NAMES].to_dict()
    healthy_patient = healthy_row[FEATURE_NAMES].to_dict()

    _, disease_proba = predict_patient(quick_model, disease_patient)
    _, healthy_proba = predict_patient(quick_model, healthy_patient)

    assert disease_proba > healthy_proba


def test_predict_patient_handles_missing_value(clean_df, quick_model):
    patient = clean_df.iloc[0][FEATURE_NAMES].to_dict()
    patient["ca"] = np.nan
    label, proba = predict_patient(quick_model, patient)
    assert label in ("Heart Disease Likely", "No Heart Disease Indicated")
    assert 0.0 <= proba <= 1.0


def test_predict_patient_threshold_changes_label():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.4, 0.6]])
    patient = {f: 0.0 for f in FEATURE_NAMES}

    label_default, _ = predict_patient(model, patient, threshold=0.5)
    label_strict, _ = predict_patient(model, patient, threshold=0.9)

    assert label_default == "Heart Disease Likely"
    assert label_strict == "No Heart Disease Indicated"
