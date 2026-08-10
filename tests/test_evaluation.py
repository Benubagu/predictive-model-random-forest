
import numpy as np
import pytest

from evaluation import (
    bootstrap_ci, build_pipeline, evaluate_hypotheses, nested_cv_compare,
    nested_cv_evaluate, select_operating_threshold,
)


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


def test_evaluate_hypotheses_h1_supported_at_both_levels():
    summary = {"accuracy": {"mean": 0.90, "std": 0.02}}
    ci = {"accuracy": {"ci_low": 0.85, "ci_high": 0.95}}
    result = evaluate_hypotheses(
        summary, ci, perm_importance_mean=[0.1, 0.0], perm_importance_std=[0.01, 0.01],
        feature_names=["a", "b"], accuracy_target=0.80, outer_folds=5,
    )
    h1 = result["h1_accuracy_at_least_target"]
    assert h1["point_estimate_meets_target"] is True
    assert h1["ci_lower_bound_meets_target"] is True


def test_evaluate_hypotheses_h1_marginal_when_ci_dips_below_target():
    # Mirrors the actual project result: point estimate clears 80% but the
    # bootstrap CI lower bound does not -- must be reported, not hidden.
    summary = {"accuracy": {"mean": 0.835, "std": 0.039}}
    ci = {"accuracy": {"ci_low": 0.789, "ci_high": 0.875}}
    result = evaluate_hypotheses(
        summary, ci, perm_importance_mean=[0.1, 0.0], perm_importance_std=[0.01, 0.01],
        feature_names=["a", "b"], accuracy_target=0.80, outer_folds=5,
    )
    h1 = result["h1_accuracy_at_least_target"]
    assert h1["point_estimate_meets_target"] is True
    assert h1["ci_lower_bound_meets_target"] is False
    assert "not strictly" in h1["conclusion"]


def test_evaluate_hypotheses_h1_not_supported_below_target():
    summary = {"accuracy": {"mean": 0.70, "std": 0.05}}
    ci = {"accuracy": {"ci_low": 0.60, "ci_high": 0.80}}
    result = evaluate_hypotheses(
        summary, ci, perm_importance_mean=[0.1, 0.0], perm_importance_std=[0.01, 0.01],
        feature_names=["a", "b"], accuracy_target=0.80, outer_folds=5,
    )
    h1 = result["h1_accuracy_at_least_target"]
    assert h1["point_estimate_meets_target"] is False
    assert "not supported" in h1["conclusion"]


def test_evaluate_hypotheses_h2_separates_signal_from_noise():
    summary = {"accuracy": {"mean": 0.90, "std": 0.02}}
    ci = {"accuracy": {"ci_low": 0.85, "ci_high": 0.95}}
    result = evaluate_hypotheses(
        summary, ci,
        perm_importance_mean=[0.06, 0.04, 0.001, -0.002],
        perm_importance_std=[0.02, 0.01, 0.01, 0.02],
        feature_names=["ca", "cp", "fbs", "age"],
        accuracy_target=0.80, outer_folds=5,
    )
    h2 = result["h2_features_differ_in_contribution"]
    assert h2["features_exceeding_1std_from_zero"] == ["ca", "cp"]
    assert set(h2["features_indistinguishable_from_zero"]) == {"fbs", "age"}


@pytest.mark.slow
def test_nested_cv_compare_default_vs_tuned_both_return_summaries(features_target):
    """RQ3 (effect of hyperparameter tuning) is answered by comparing a
    default-hyperparameter RF against a tuned one under nested_cv_compare,
    the same mechanism tuning_effect.py uses. An empty param_grid={} is the
    "default hyperparameters" case -- this is the first test to exercise
    that path, since every other nested_cv_compare use passes a real grid."""
    X, y = features_target
    default_result = nested_cv_compare(
        X, y, build_pipeline, param_grid={}, outer_folds=3, inner_folds=2, random_state=42,
    )
    tuned_result = nested_cv_compare(
        X, y, build_pipeline, param_grid={"rf__n_estimators": [100], "rf__max_depth": [5]},
        outer_folds=3, inner_folds=2, random_state=42,
    )
    for result in (default_result, tuned_result):
        assert set(result["summary"]) == {"accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"}
        assert 0.0 <= result["summary"]["accuracy"]["mean"] <= 1.0
    # default fold_metrics' best_params should be empty (no tuning happened)
    assert all(fm["best_params"] == {} for fm in default_result["fold_metrics"])


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
