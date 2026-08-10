"""
config.py
---------
Centralized paths and constants for the heart disease Random Forest
system. train_model.py, evaluation.py, and predict.py all import from
here so they never disagree on where data, models, or outputs live.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "heart.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "random_forest_heart_model.joblib"
THRESHOLD_PATH = MODEL_DIR / "operating_threshold.json"
OUTPUT_DIR = PROJECT_ROOT / "output"

RANDOM_STATE = 42

# Nested cross-validation: the outer loop produces the honest, unbiased
# performance estimate; the inner loop (run independently inside each
# outer training fold) is where hyperparameter search happens. The same
# INNER_CV_FOLDS value is reused for the final full-data refit and for
# CalibratedClassifierCV's internal folds.
OUTER_CV_FOLDS = 5
INNER_CV_FOLDS = 5

N_BOOTSTRAP = 1000
CALIBRATION_METHOD = "sigmoid"  # Platt scaling; safer than isotonic at n=303
PERMUTATION_REPEATS = 20

# Minimum sensitivity (recall on the disease class) the chosen decision
# threshold must guarantee. In a screening context a false negative
# (missed disease) is far costlier than a false alarm, so we pick the
# highest threshold that still clears this floor rather than defaulting
# to 0.5. See evaluation.select_operating_threshold.
TARGET_SENSITIVITY = 0.95

# Grid search space for the Pipeline's "rf" step (see evaluation.build_pipeline).
PARAM_GRID = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [None, 5, 10],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
}
