"""
config.py
---------
Centralized paths and constants for the heart disease Random Forest
system. train_model.py and predict.py both import from here so they
never disagree on where data, models, or outputs live.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "heart.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "random_forest_heart_model.joblib"
OUTPUT_DIR = PROJECT_ROOT / "output"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Grid search space for the Pipeline's "rf" step (see train_model.build_pipeline).
PARAM_GRID = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [None, 5, 10],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
}
