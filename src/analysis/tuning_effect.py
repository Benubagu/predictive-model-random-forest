"""
src/analysis/tuning_effect.py
--------------------------------
Answers RQ3 ("to what extent does hyperparameter tuning affect the
predictive performance of the Random Forest classifier?") directly:
runs the *same* Random Forest pipeline under the *same* nested CV
protocol (evaluation.nested_cv_compare, same folds, same seed) twice --
once with scikit-learn's default hyperparameters, once with the tuned
grid search space (config.PARAM_GRID) -- and reports the delta.

This stays within the thesis's single-algorithm scope (Sec. 1.7): both
sides of the comparison are Random Forest, differing only in
hyperparameters, not in algorithm.

Run from the repository root:
    python -m src.analysis.tuning_effect
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core import config
from src.core.evaluation import build_pipeline, nested_cv_compare
from src.core.preprocessing import clean_data, get_features_and_target, load_raw_data

logger = logging.getLogger(__name__)


def plot_tuning_effect(default_summary, tuned_summary, out_path):
    metrics = ["accuracy", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "F1", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 5))
    default_means = [default_summary[m]["mean"] for m in metrics]
    default_stds = [default_summary[m]["std"] for m in metrics]
    tuned_means = [tuned_summary[m]["mean"] for m in metrics]
    tuned_stds = [tuned_summary[m]["std"] for m in metrics]

    ax.bar(x - width / 2, default_means, width, yerr=default_stds, capsize=3,
           label="Default hyperparameters", color="#94a3b8")
    ax.bar(x + width / 2, tuned_means, width, yerr=tuned_stds, capsize=3,
           label="Tuned (grid search)", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Nested CV score (mean +/- std across outer folds)")
    ax.set_title("Effect of hyperparameter tuning (RQ3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--outer-folds", type=int, default=config.OUTER_CV_FOLDS)
    parser.add_argument("--inner-folds", type=int, default=config.INNER_CV_FOLDS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(args.data)
    df = clean_data(raw)
    X, y = get_features_and_target(df)

    logger.info("Running nested CV for default-hyperparameter Random Forest...")
    default_result = nested_cv_compare(
        X, y, build_pipeline, param_grid={}, outer_folds=args.outer_folds,
        inner_folds=args.inner_folds, random_state=args.seed,
    )
    for metric, stats in default_result["summary"].items():
        logger.info("  default %-10s: %.4f +/- %.4f", metric, stats["mean"], stats["std"])

    logger.info("Running nested CV for tuned (grid search) Random Forest...")
    tuned_result = nested_cv_compare(
        X, y, build_pipeline, param_grid=config.PARAM_GRID, outer_folds=args.outer_folds,
        inner_folds=args.inner_folds, random_state=args.seed,
    )
    for metric, stats in tuned_result["summary"].items():
        logger.info("  tuned   %-10s: %.4f +/- %.4f", metric, stats["mean"], stats["std"])

    delta = {
        metric: tuned_result["summary"][metric]["mean"] - default_result["summary"][metric]["mean"]
        for metric in default_result["summary"]
    }
    logger.info("=== Delta (tuned - default) ===")
    for metric, d in delta.items():
        logger.info("  %-10s: %+.4f", metric, d)

    with open(args.output / "tuning_effect.json", "w") as f:
        json.dump({
            "default": default_result,
            "tuned": tuned_result,
            "delta_tuned_minus_default": delta,
        }, f, indent=2)

    plot_tuning_effect(default_result["summary"], tuned_result["summary"], args.output / "tuning_effect.png")

    logger.info("Saved to %s and %s", args.output / "tuning_effect.json", args.output / "tuning_effect.png")


if __name__ == "__main__":
    main()
