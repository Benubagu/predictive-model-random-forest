"""
train_model.py
----------------
Trains and evaluates a Random Forest classifier for heart disease
diagnosis, tunes hyperparameters via grid search, and saves the
final pipeline plus evaluation artifacts (metrics, confusion matrix,
ROC curve, feature importance chart).

Imputation and the classifier are combined into a single sklearn
Pipeline so the imputer is refit on each training fold during grid
search instead of once on the full dataset beforehand — this avoids
train/test leakage through the imputation statistics.
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

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

import config
from preprocessing import load_raw_data, clean_data, get_features_and_target, FEATURE_NAMES

logger = logging.getLogger(__name__)

TARGET_NAMES = ["No Disease", "Disease"]


def build_pipeline(random_state):
    """Imputation + Random Forest, fit together so imputation stays inside CV."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(random_state=random_state, class_weight="balanced")),
    ])


def split_data(X, y, test_size, random_state):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def tune_pipeline(X_train, y_train, param_grid, cv_folds, random_state):
    """Grid search over the imputation + Random Forest pipeline."""
    pipe = build_pipeline(random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
    cm = confusion_matrix(y_test, y_pred)
    return metrics, report, cm, y_pred, y_proba


def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(TARGET_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(TARGET_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - Random Forest")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_test, y_proba, auc, out_path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.3f})", color="#2563eb")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(rf, feature_names, out_path):
    importances = rf.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(feature_names)[order], importances[order], color="#2563eb")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance - Random Forest")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.DATA_PATH,
                         help="Path to the input CSV (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR,
                         help="Directory to write evaluation artifacts to (default: %(default)s)")
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH,
                         help="Path to save the trained pipeline to (default: %(default)s)")
    parser.add_argument("--cv-folds", type=int, default=config.CV_FOLDS,
                         help="Number of stratified CV folds used during grid search (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE,
                         help="Random seed for the split and the model (default: %(default)s)")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE,
                         help="Held-out test set fraction (default: %(default)s)")
    parser.add_argument("--no-plots", action="store_true",
                         help="Skip generating plot artifacts (confusion matrix, ROC curve, feature importance)")
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
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.seed)

    logger.info("Training samples: %d, Test samples: %d", len(X_train), len(X_test))
    logger.info("Running grid search for best pipeline hyperparameters...")
    best_pipeline, best_params, best_cv_f1 = tune_pipeline(
        X_train, y_train, config.PARAM_GRID, args.cv_folds, args.seed
    )
    logger.info("Best parameters: %s", best_params)
    logger.info("Best cross-validated F1 score: %.4f", best_cv_f1)

    metrics, report, cm, _, y_proba = evaluate_model(best_pipeline, X_test, y_test)

    logger.info("=== Test Set Performance ===")
    for k, v in metrics.items():
        logger.info("%-12s: %.4f", k, v)
    logger.info("\n%s", report)

    joblib.dump(best_pipeline, args.model_path)

    with open(args.output / "metrics.json", "w") as f:
        json.dump({
            "best_params": best_params,
            "best_cv_f1": best_cv_f1,
            "test_metrics": metrics,
        }, f, indent=2)

    with open(args.output / "classification_report.txt", "w") as f:
        f.write(report)

    if not args.no_plots:
        plot_confusion_matrix(cm, args.output / "confusion_matrix.png")
        plot_roc_curve(y_test, y_proba, metrics["roc_auc"], args.output / "roc_curve.png")
        plot_feature_importance(best_pipeline.named_steps["rf"], FEATURE_NAMES, args.output / "feature_importance.png")

    logger.info("Model saved to %s", args.model_path)
    logger.info("Evaluation artifacts saved to %s", args.output)


if __name__ == "__main__":
    main()
