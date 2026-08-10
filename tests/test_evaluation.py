
import numpy as np
import pytest

from evaluation import bootstrap_ci, build_pipeline, nested_cv_evaluate, select_operating_threshold


def test_select_operating_threshold_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold, sensitivity, specificity = select_operating_threshold(
        y_true, y_proba, target_sensitivity=1.0
    )
    assert sensitivity == 1.0
    assert specificity == 1.0
    assert 0.3 < threshold <= 0.7


def test_select_operating_threshold_predicts_all_positive_when_thats_the_only_way():
    # target_sensitivity=1.0 is always achievable by predicting everyone
    # positive (threshold = min probability); with an inverted model that's
    # also the *only* way to catch the single true positive.
    y_true = np.array([0, 1])
    y_proba = np.array([0.9, 0.1])  # inverted: the diseased patient scores lower
    threshold, sensitivity, specificity = select_operating_threshold(
        y_true, y_proba, target_sensitivity=1.0
    )
    assert threshold == pytest.approx(0.1)
    assert sensitivity == 1.0
    assert specificity == 0.0


def test_bootstrap_ci_on_perfect_predictions_is_tight_at_one():
    rng = np.random.default_rng(0)
    y_true = (rng.random(200) > 0.5).astype(int)
    y_proba = y_true.astype(float)  # perfect predictions
    ci = bootstrap_ci(y_true, y_proba, n_boot=200, random_state=0)
    assert ci["accuracy"]["mean"] == pytest.approx(1.0)
    assert ci["accuracy"]["ci_low"] == pytest.approx(1.0, abs=1e-9)


def test_build_pipeline_has_impute_then_rf_steps():
    pipe = build_pipeline(random_state=0)
    assert list(pipe.named_steps) == ["impute", "rf"]


@pytest.mark.slow
def test_nested_cv_regression_band(features_target):
    """Fixed-seed nested CV with a reduced grid/fold count for speed. If
    this starts failing, something in the pipeline wiring (features,
    target, imputation, CV splits) silently broke -- the exact class of
    bug a single manual run won't catch."""
    X, y = features_target
    small_grid = {"rf__n_estimators": [100], "rf__max_depth": [5]}
    result = nested_cv_evaluate(
        X, y, feature_names=list(X.columns), param_grid=small_grid,
        outer_folds=3, inner_folds=2, random_state=42,
        calibration_method="sigmoid", perm_repeats=5,
    )
    accuracy = result["summary"]["accuracy"]["mean"]
    roc_auc = result["summary"]["roc_auc"]["mean"]
    assert 0.70 <= accuracy <= 0.95
    assert roc_auc >= 0.80
