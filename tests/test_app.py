import pytest

pytest.importorskip("streamlit")  # app.py is optional (requirements-app.txt); skip cleanly if absent

import numpy as np

import app
from src.core.evaluation import build_pipeline
from src.core.preprocessing import FEATURE_NAMES
from src.inference import predict


def test_app_reuses_predict_module_functions_without_reimplementation():
    # The whole point of app.py is a thin UI layer -- these must be the
    # exact same function objects as predict.py, not lookalike copies that
    # could silently diverge in behavior.
    assert app.load_model is predict.load_model
    assert app.load_threshold is predict.load_threshold
    assert app.predict_patient is predict.predict_patient


def test_app_categorical_and_continuous_features_partition_all_features():
    assert sorted(app.CATEGORICAL_FEATURES + app.CONTINUOUS_FEATURES) == sorted(FEATURE_NAMES)
    assert set(app.CATEGORICAL_FEATURES).isdisjoint(app.CONTINUOUS_FEATURES)


def test_app_defaults_cover_every_feature():
    assert set(app.DEFAULTS) == set(FEATURE_NAMES)


@pytest.fixture(scope="module")
def quick_model(clean_df):
    X = clean_df[FEATURE_NAMES]
    y = clean_df["target"]
    pipe = build_pipeline(random_state=42)
    pipe.fit(X, y)
    return pipe


def test_app_default_patient_predicts_via_shared_predict_patient(quick_model):
    # Exercises app.py's own DEFAULTS dict through the real predict_patient
    # (identity-checked above), confirming the default form state doesn't
    # crash and produces a well-formed prediction.
    patient = dict(app.DEFAULTS)
    label, proba = app.predict_patient(quick_model, patient)
    assert label in ("Heart Disease Likely", "No Heart Disease Indicated")
    assert 0.0 <= proba <= 1.0


def test_app_unknown_categorical_becomes_nan_like_predict_py(quick_model):
    patient = dict(app.DEFAULTS)
    patient["thal"] = np.nan  # what app.py sends when the user picks "Unknown"
    label, proba = app.predict_patient(quick_model, patient)
    assert label in ("Heart Disease Likely", "No Heart Disease Indicated")
    assert 0.0 <= proba <= 1.0
