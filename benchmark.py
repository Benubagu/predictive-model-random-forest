"""
benchmark.py
-------------
Compares the production Random Forest pipeline against two simpler
baselines -- Logistic Regression and a single Decision Tree -- under the
same nested cross-validation protocol (config.OUTER_CV_FOLDS /
INNER_CV_FOLDS, same seed). Same data, same CV splits, same scoring:
whatever gap shows up between Random Forest and Logistic Regression is a
real gap, not an artifact of unequal evaluation, and Logistic Regression
being competitive would be a legitimate reason to prefer its
interpretability instead.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from evaluation import build_baseline_pipeline, build_pipeline, nested_cv_compare
from preprocessing import clean_data, get_features_and_target, load_raw_data

logger = logging.getLogger(__name__)

MODELS = {
    "random_forest": (build_pipeline, config.PARAM_GRID),
    "logistic_regression": (
        lambda rs: build_baseline_pipeline("logreg", rs),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "decision_tree": (
        lambda rs: build_baseline_pipeline("dtree", rs),
        {"clf__max_depth": [3, 5, 10, None], "clf__min_samples_leaf": [1, 2, 5]},
    ),
}

DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
}


def plot_comparison(summaries, out_path):
    metrics = ["accuracy", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "F1", "ROC-AUC"]
    model_names = list(summaries.keys())
    colors = ["#2563eb", "#f97316", "#16a34a"]

    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(model_names):
        means = [summaries[name][m]["mean"] for m in metrics]
        stds = [summaries[name][m]["std"] for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
               label=DISPLAY_NAMES[name], color=colors[i % len(colors)])
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Nested CV score (mean +/- std across outer folds)")
    ax.set_title("Random Forest vs. baselines")
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

    results = {}
    for name, (pipeline_builder, param_grid) in MODELS.items():
        logger.info("Running nested CV for %s...", DISPLAY_NAMES[name])
        result = nested_cv_compare(
            X, y, pipeline_builder, param_grid, args.outer_folds, args.inner_folds, args.seed,
        )
        results[name] = result
        for metric, stats in result["summary"].items():
            logger.info("  %-10s: %.4f +/- %.4f", metric, stats["mean"], stats["std"])

    with open(args.output / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_comparison({name: r["summary"] for name, r in results.items()}, args.output / "model_comparison.png")

    logger.info("Comparison saved to %s", args.output / "model_comparison.json")
    logger.info("Plot saved to %s", args.output / "model_comparison.png")


if __name__ == "__main__":
    main()
