"""
src/training/train_model.py
-----------------------------
Trains and evaluates a Random Forest classifier for heart disease
diagnosis.

Run from the repository root as a module:
    python -m src.training.train_model

Methodology (see src/core/evaluation.py for the "why"):
  1. Nested cross-validation gives an honest, low-variance estimate of
     generalization performance (mean +/- std across outer folds) plus
     bootstrap confidence intervals, computed from leakage-free
     out-of-fold predictions pooled across the whole dataset.
  2. A decision threshold is chosen (not the default 0.5) to guarantee
     a minimum sensitivity, since a missed heart-disease case is far
     costlier than a false alarm in a screening context.
  3. Permutation importance (computed per outer test fold) complements
     the Random Forest's impurity-based importances, which are biased
     toward high-cardinality continuous features.
  4. The final deployed model is a fresh pipeline, retuned and refit on
     the FULL dataset, then wrapped in CalibratedClassifierCV so its
     predicted probabilities are trustworthy, not just its rankings.
     This refit is standard practice for deployment — nested CV already
     produced the honest performance estimate independently of it.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.core import config
from src.core.evaluation import (
    build_pipeline, bootstrap_ci, evaluate_hypotheses, nested_cv_evaluate, select_operating_threshold,
)
from src.core.preprocessing import load_raw_data, clean_data, get_features_and_target, FEATURE_NAMES

logger = logging.getLogger(__name__)

TARGET_NAMES = ["No Disease", "Disease"]


def plot_confusion_matrix(cm, out_path, title="Confusion Matrix - Random Forest"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(TARGET_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(TARGET_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, auc, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.3f})", color="#2563eb")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (pooled out-of-fold predictions)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true, y_proba, pr_auc, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    prevalence = y_true.mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"Random Forest (PR-AUC = {pr_auc:.3f})", color="#2563eb")
    ax.axhline(prevalence, linestyle="--", color="gray", label=f"Chance (prevalence = {prevalence:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (pooled out-of-fold predictions)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration_curve(y_true, proba_raw, proba_calibrated, out_path):
    frac_raw, mean_raw = calibration_curve(y_true, proba_raw, n_bins=10, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_true, proba_calibrated, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(mean_raw, frac_raw, "o-", color="#f97316", label="Raw pipeline")
    ax.plot(mean_cal, frac_cal, "o-", color="#2563eb", label="Calibrated (sigmoid)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction with disease")
    ax.set_title("Calibration Curve (pooled out-of-fold predictions)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_importance(importances, feature_names, title, out_path, xerr=None):
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 5))
    err = None if xerr is None else np.asarray(xerr)[order]
    ax.barh(np.array(feature_names)[order], np.asarray(importances)[order],
            xerr=err, color="#2563eb")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fit_final_model(X, y, param_grid, inner_folds, random_state, calibration_method):
    """Retune on the full dataset and return a calibrated deployment pipeline
    plus the uncalibrated reference pipeline (used for impurity importances)."""
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=build_pipeline(random_state),
        param_grid=param_grid,
        cv=inner_cv,
        scoring="f1",
        n_jobs=-1,
    )
    search.fit(X, y)

    reference_pipeline = build_pipeline(random_state).set_params(**search.best_params_)
    reference_pipeline.fit(X, y)

    calibrated = CalibratedClassifierCV(
        estimator=build_pipeline(random_state).set_params(**search.best_params_),
        method=calibration_method,
        cv=inner_cv,
    )
    calibrated.fit(X, y)

    return calibrated, reference_pipeline, search.best_params_


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.DATA_PATH,
                         help="Path to the input CSV (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR,
                         help="Directory to write evaluation artifacts to (default: %(default)s)")
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH,
                         help="Path to save the trained pipeline to (default: %(default)s)")
    parser.add_argument("--outer-folds", type=int, default=config.OUTER_CV_FOLDS,
                         help="Outer CV folds for the performance estimate (default: %(default)s)")
    parser.add_argument("--inner-folds", type=int, default=config.INNER_CV_FOLDS,
                         help="Inner CV folds for hyperparameter tuning/calibration (default: %(default)s)")
    parser.add_argument("--n-bootstrap", type=int, default=config.N_BOOTSTRAP,
                         help="Bootstrap resamples for confidence intervals (default: %(default)s)")
    parser.add_argument("--target-sensitivity", type=float, default=config.TARGET_SENSITIVITY,
                         help="Minimum sensitivity the operating threshold must guarantee (default: %(default)s)")
    parser.add_argument("--calibration-method", choices=["sigmoid", "isotonic"],
                         default=config.CALIBRATION_METHOD,
                         help="Probability calibration method (default: %(default)s)")
    parser.add_argument("--perm-repeats", type=int, default=config.PERMUTATION_REPEATS,
                         help="Permutation repeats per outer fold (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE,
                         help="Random seed (default: %(default)s)")
    parser.add_argument("--no-plots", action="store_true",
                         help="Skip generating plot artifacts")
    parser.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="Logging verbosity (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(args.data)
    df = clean_data(raw)
    X, y = get_features_and_target(df)

    logger.info("Dataset: %d patients, %d features", len(X), X.shape[1])
    logger.info("Running nested cross-validation (outer=%d, inner=%d)...", args.outer_folds, args.inner_folds)
    cv_result = nested_cv_evaluate(
        X, y, FEATURE_NAMES, config.PARAM_GRID, args.outer_folds, args.inner_folds,
        args.seed, args.calibration_method, args.perm_repeats,
    )

    logger.info("=== Nested CV performance (mean +/- std across %d outer folds) ===", args.outer_folds)
    for metric, stats in cv_result["summary"].items():
        logger.info("%-10s: %.4f +/- %.4f", metric, stats["mean"], stats["std"])

    logger.info("Bootstrapping %d-sample confidence intervals on pooled out-of-fold predictions...",
                args.n_bootstrap)
    ci_raw = bootstrap_ci(cv_result["oof_true"], cv_result["oof_proba_raw"], args.n_bootstrap, args.seed)
    for metric, stats in ci_raw.items():
        logger.info("%-10s: %.4f  95%% CI [%.4f, %.4f]", metric, stats["mean"], stats["ci_low"], stats["ci_high"])

    threshold, sensitivity, specificity = select_operating_threshold(
        cv_result["oof_true"], cv_result["oof_proba_calibrated"], args.target_sensitivity,
    )
    logger.info("Operating threshold (calibrated probabilities, target sensitivity >= %.2f): %.4f",
                args.target_sensitivity, threshold)
    logger.info("  -> sensitivity=%.4f  specificity=%.4f", sensitivity, specificity)

    y_pred_operating = (cv_result["oof_proba_calibrated"] >= threshold).astype(int)
    operating_report = classification_report(
        cv_result["oof_true"], y_pred_operating, target_names=TARGET_NAMES,
    )
    operating_cm = confusion_matrix(cv_result["oof_true"], y_pred_operating)
    logger.info("\n%s", operating_report)

    logger.info("Refitting final pipeline on the full dataset for deployment...")
    final_model, reference_pipeline, best_params_full = fit_final_model(
        X, y, config.PARAM_GRID, args.inner_folds, args.seed, args.calibration_method,
    )
    logger.info("Final model hyperparameters: %s", best_params_full)

    joblib.dump(final_model, args.model_path)
    with open(config.THRESHOLD_PATH, "w") as f:
        json.dump({
            "threshold": threshold,
            "target_sensitivity": args.target_sensitivity,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }, f, indent=2)

    impurity_importance = reference_pipeline.named_steps["rf"].feature_importances_
    top_impurity = np.array(FEATURE_NAMES)[np.argsort(impurity_importance)[::-1][:5]]
    top_permutation = np.array(FEATURE_NAMES)[np.argsort(cv_result["perm_importance_mean"])[::-1][:5]]
    logger.info("Top 5 features by impurity importance:    %s", list(top_impurity))
    logger.info("Top 5 features by permutation importance: %s", list(top_permutation))

    hypothesis_tests = evaluate_hypotheses(
        cv_result["summary"], ci_raw, cv_result["perm_importance_mean"], cv_result["perm_importance_std"],
        FEATURE_NAMES, config.ACCURACY_TARGET, args.outer_folds,
    )
    logger.info("=== Hypothesis tests ===")
    logger.info(hypothesis_tests["h1_accuracy_at_least_target"]["conclusion"])
    logger.info(hypothesis_tests["h2_features_differ_in_contribution"]["conclusion"])

    with open(args.output / "metrics.json", "w") as f:
        json.dump({
            "n_patients": len(X),
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "fold_metrics": cv_result["fold_metrics"],
            "nested_cv_summary": cv_result["summary"],
            "bootstrap_ci": ci_raw,
            "operating_point": {
                "threshold": threshold,
                "target_sensitivity": args.target_sensitivity,
                "sensitivity": sensitivity,
                "specificity": specificity,
            },
            "final_model_params": best_params_full,
            "calibration_method": args.calibration_method,
            "impurity_importance": dict(zip(FEATURE_NAMES, impurity_importance.tolist())),
            "permutation_importance_mean": dict(zip(FEATURE_NAMES, cv_result["perm_importance_mean"].tolist())),
            "permutation_importance_std": dict(zip(FEATURE_NAMES, cv_result["perm_importance_std"].tolist())),
            "hypothesis_tests": hypothesis_tests,
        }, f, indent=2)

    with open(args.output / "classification_report.txt", "w") as f:
        f.write(f"Classification report at operating threshold {threshold:.4f}\n")
        f.write("(calibrated probabilities, pooled out-of-fold predictions)\n\n")
        f.write(operating_report)

    if not args.no_plots:
        plot_confusion_matrix(
            operating_cm, args.output / "confusion_matrix.png",
            title=f"Confusion Matrix (threshold={threshold:.3f})",
        )
        plot_roc_curve(cv_result["oof_true"], cv_result["oof_proba_raw"],
                        cv_result["summary"]["roc_auc"]["mean"], args.output / "roc_curve.png")
        plot_pr_curve(cv_result["oof_true"], cv_result["oof_proba_raw"],
                      cv_result["summary"]["pr_auc"]["mean"], args.output / "pr_curve.png")
        plot_calibration_curve(cv_result["oof_true"], cv_result["oof_proba_raw"],
                                cv_result["oof_proba_calibrated"], args.output / "calibration_curve.png")
        plot_importance(impurity_importance, FEATURE_NAMES, "Feature Importance - Impurity (MDI)",
                         args.output / "feature_importance.png")
        plot_importance(cv_result["perm_importance_mean"], FEATURE_NAMES,
                         "Feature Importance - Permutation (mean F1 drop)",
                         args.output / "permutation_importance.png",
                         xerr=cv_result["perm_importance_std"])

    logger.info("Model saved to %s", args.model_path)
    logger.info("Operating threshold saved to %s", config.THRESHOLD_PATH)
    logger.info("Evaluation artifacts saved to %s", args.output)


if __name__ == "__main__":
    main()
