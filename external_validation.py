"""
external_validation.py
------------------------
Scores the trained Cleveland model against the Hungarian, Switzerland,
and VA Long Beach cohorts -- the same 13 UCI attributes, collected at
three different sites with different equipment and data entry practices.
This is a genuine generalization test: the model, its imputer, and its
calibration were all fit exclusively on Cleveland data in train_model.py,
so nothing about these cohorts' own distribution influences how they are
processed here.

Run after train_model.py has produced model/random_forest_heart_model.joblib.

Known data quality note: Switzerland (all 123 rows) and VA Long Beach
(49/200 rows) encode missing cholesterol as a literal 0 rather than '?'.
preprocessing.clean_data() treats trestbps/chol == 0 as missing for this
reason (see its docstring) -- Cleveland has no such rows, so this doesn't
change anything about the training data.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

import config
from evaluation import compute_metrics
from preprocessing import clean_data, get_features_and_target, load_raw_data

logger = logging.getLogger(__name__)

COHORT_LABELS = {"hungarian": "Hungarian", "switzerland": "Switzerland", "va": "VA Long Beach"}


def load_cohort(name):
    path = config.EXTERNAL_DATA_DIR / f"{name}.csv"
    raw = load_raw_data(path, validate=False)
    clean = clean_data(raw)
    return get_features_and_target(clean)


def load_operating_threshold(path=config.THRESHOLD_PATH, default=0.5):
    try:
        with open(path) as f:
            return json.load(f)["threshold"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        logger.warning("No operating threshold found at %s, using %.2f", path, default)
        return default


def specificity_at(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def evaluate_cohort(model, X, y, threshold):
    y_true = y.to_numpy()
    y_proba = model.predict_proba(X)[:, 1]
    result = {
        "n_patients": len(y),
        "prevalence": float(y.mean()),
        "metrics_at_0.5": compute_metrics(y_true, y_proba, threshold=0.5),
        "metrics_at_operating_threshold": {
            "threshold": threshold,
            "specificity": specificity_at(y_true, y_proba, threshold),
            **compute_metrics(y_true, y_proba, threshold=threshold),
        },
    }
    return result, y_true, y_proba


def plot_roc_curves(curves, out_path):
    """curves: dict of {label: (y_true, y_proba)}"""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    colors = ["#2563eb", "#f97316", "#16a34a", "#dc2626"]
    for (label, (y_true, y_proba)), color in zip(curves.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = compute_metrics(y_true, y_proba)["roc_auc"]
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: Cleveland-trained model on external cohorts")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model_path)
    threshold = load_operating_threshold()

    results = {}
    curves = {}
    X_parts, y_parts = [], []

    for name in config.EXTERNAL_COHORTS:
        X, y = load_cohort(name)
        X_parts.append(X)
        y_parts.append(y)
        result, y_true, y_proba = evaluate_cohort(model, X, y, threshold)
        results[name] = result
        curves[COHORT_LABELS[name]] = (y_true, y_proba)
        logger.info(
            "%-12s n=%3d  prevalence=%.2f  roc_auc=%.4f  accuracy@0.5=%.4f  sensitivity@op=%.4f  specificity@op=%.4f",
            name, result["n_patients"], result["prevalence"],
            result["metrics_at_0.5"]["roc_auc"],
            result["metrics_at_0.5"]["accuracy"],
            result["metrics_at_operating_threshold"]["recall"],
            result["metrics_at_operating_threshold"]["specificity"],
        )

    X_combined = pd.concat(X_parts, ignore_index=True)
    y_combined = pd.concat(y_parts, ignore_index=True)
    combined_result, y_true, y_proba = evaluate_cohort(model, X_combined, y_combined, threshold)
    results["combined"] = combined_result
    curves["All external (n=617)"] = (y_true, y_proba)
    logger.info(
        "%-12s n=%3d  prevalence=%.2f  roc_auc=%.4f  accuracy@0.5=%.4f",
        "combined", combined_result["n_patients"], combined_result["prevalence"],
        combined_result["metrics_at_0.5"]["roc_auc"], combined_result["metrics_at_0.5"]["accuracy"],
    )

    with open(args.output / "external_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_roc_curves(curves, args.output / "external_validation_roc.png")

    logger.info("Results saved to %s", args.output / "external_validation.json")
    logger.info("ROC plot saved to %s", args.output / "external_validation_roc.png")


if __name__ == "__main__":
    main()
