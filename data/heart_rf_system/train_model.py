"""
train_model.py
----------------
Trains and evaluates a Random Forest classifier for heart disease
diagnosis, tunes hyperparameters via grid search, and saves the
final model plus evaluation artifacts (metrics, confusion matrix,
ROC curve, feature importance chart).
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from preprocessing import load_raw_data, clean_data, get_features_and_target, FEATURE_NAMES

RANDOM_STATE = 42


def split_data(X, y, test_size=0.2):
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def tune_random_forest(X_train, y_train):
    """Grid search over key Random Forest hyperparameters."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=rf,
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
    report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])
    cm = confusion_matrix(y_test, y_pred)
    return metrics, report, cm, y_pred, y_proba


def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Disease", "Disease"])
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


def plot_feature_importance(model, feature_names, out_path):
    importances = model.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(feature_names)[order], importances[order], color="#2563eb")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance - Random Forest")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    raw = load_raw_data("data/heart.csv")
    df = clean_data(raw)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print("Running grid search for best Random Forest hyperparameters...")
    best_model, best_params, best_cv_f1 = tune_random_forest(X_train, y_train)
    print("Best parameters:", best_params)
    print(f"Best cross-validated F1 score: {best_cv_f1:.4f}")

    metrics, report, cm, y_pred, y_proba = evaluate_model(best_model, X_test, y_test)

    print("\n=== Test Set Performance ===")
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")
    print("\n" + report)

    # Save model
    joblib.dump(best_model, "model/random_forest_heart_model.joblib")

    # Save metrics
    with open("output/metrics.json", "w") as f:
        json.dump({
            "best_params": best_params,
            "best_cv_f1": best_cv_f1,
            "test_metrics": metrics,
        }, f, indent=2)

    with open("output/classification_report.txt", "w") as f:
        f.write(report)

    # Save plots
    plot_confusion_matrix(cm, "output/confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, metrics["roc_auc"], "output/roc_curve.png")
    plot_feature_importance(best_model, FEATURE_NAMES, "output/feature_importance.png")

    print("\nModel and evaluation artifacts saved to model/ and output/")


if __name__ == "__main__":
    main()
