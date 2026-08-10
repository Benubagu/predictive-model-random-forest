# Predictive Model for Heart Disease Diagnosis Using Random Forest Algorithm

A complete, working machine learning system that diagnoses the likely presence
of heart disease from patient clinical data using the Random Forest algorithm.

## Dataset

Cleveland Heart Disease dataset (UCI Machine Learning Repository) — 303 patient
records, 13 clinical attributes. The original multi-class severity label
(0-4) is collapsed into a binary target: **0 = no disease, 1 = disease present**.

| Attribute | Description |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (1-4) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1/0) |
| restecg | Resting ECG results (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1/0) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment (1-3) |
| ca | Number of major vessels colored by fluoroscopy (0-3) |
| thal | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) |

## Project structure

```
.
├── data/
│   └── heart.csv                  # Raw dataset
├── docs/
│   └── data_dictionary.md          # Attribute reference + data quality notes
├── model/                          # Trained pipeline + threshold (created by train_model.py, gitignored)
├── output/                         # Metrics + plots (created by train_model.py, gitignored)
├── tests/                           # pytest regression suite
├── config.py                       # Paths, seeds, CV folds, param grid
├── preprocessing.py                # Data loading, cleaning, integrity check
├── evaluation.py                   # Nested CV, bootstrap CIs, threshold selection
├── train_model.py                  # Orchestrates training + evaluation, saves artifacts
├── predict.py                      # CLI diagnosis tool for new patients
├── run.sh                          # One command: clean + train (or install/test/clean alone)
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt            # requirements.txt + pytest
├── .gitignore
└── README.md
```

## How it works

1. **preprocessing.py** — loads the raw CSV, converts `?` missing-value
   markers to NaN, casts columns to numeric, and derives the binary
   diagnosis label. It does **not** impute missing values: imputation is a
   fold-dependent statistic, so it lives inside the sklearn `Pipeline`
   (`SimpleImputer` → `RandomForestClassifier`) built in `evaluation.py`
   and is refit on every training fold. Nothing about a held-out patient,
   in any fold, influences their own prediction.
2. **evaluation.py** — runs nested cross-validation: an outer loop (5
   folds) produces the honest performance estimate, while hyperparameter
   tuning happens on an independent inner loop (5 folds) inside each outer
   training fold. The outer loop also produces leakage-free out-of-fold
   probability predictions for the entire dataset — both raw and
   calibrated (`CalibratedClassifierCV`, Platt/sigmoid) — plus per-fold
   permutation importance.
3. **train_model.py** — orchestrates the above, bootstraps 95% confidence
   intervals from the pooled out-of-fold predictions, picks a decision
   threshold that guarantees a minimum sensitivity (see below), refits a
   final calibrated pipeline on the full dataset for deployment, and saves
   the model, threshold, metrics, and plots.
4. **predict.py** — loads the saved pipeline and the tuned threshold, then
   diagnoses a new patient (missing fields are left blank and imputed by
   the pipeline) either interactively or via `predict_patient()`.

## Results obtained

Single train/test splits on 303 patients are too noisy to report to a
tenth of a percent — a 61-patient test set has a wide error bar. These
numbers instead come from 5-fold nested cross-validation: mean ± std
across the outer folds, with 95% bootstrap confidence intervals on the
pooled out-of-fold predictions (raw, uncalibrated pipeline; threshold 0.5).

| Metric | Mean ± std | 95% CI |
|---|---|---|
| Accuracy | 83.5% ± 3.9% | [78.9%, 87.5%] |
| Precision | 84.1% ± 4.7% | [77.7%, 90.2%] |
| Recall | 79.1% ± 7.0% | [71.8%, 85.5%] |
| F1 score | 81.4% ± 4.7% | [76.0%, 86.0%] |
| ROC-AUC | 0.908 ± 0.027 | [0.874, 0.938] |
| PR-AUC | 0.902 ± 0.037 | [0.856, 0.938] |

### Operating point

For deployment, the decision threshold is tuned (not left at 0.5) to
guarantee **≥95% sensitivity** on calibrated out-of-fold probabilities,
since a missed heart-disease case (false negative) is far costlier than a
false alarm in a screening context. This lands at threshold **0.164**,
giving **97.1% sensitivity / 51.8% specificity** — the model deliberately
over-refers borderline patients rather than risk missing disease. Exact
figures and both the impurity and permutation feature-importance rankings
are in `output/metrics.json`.

The two importance rankings disagree on `age`, `chol`, and `thalach`:
impurity importance (biased toward high-cardinality continuous features)
ranks them highly, while permutation importance — computed by measuring
the F1 drop from shuffling each feature on genuinely held-out folds — does
not. `ca`, `cp`, and `thal` are the features both methods agree matter.

## Running it

```bash
pip install -r requirements.txt

# Train (also re-generates the model + all output artifacts)
python train_model.py

# Diagnose a new patient interactively
python predict.py

# Or, one command for the above (clean + train):
./run.sh
```

`run.sh` also has `install`, `train`, `test`, and `clean` targets
(`./run.sh test` runs the pytest suite). It prefers the `python` command
over `python3` — on Windows, `python3` can resolve to the App Execution
Alias stub under `WindowsApps/`, a distinct, partially-populated Python
environment that will fail trying to build `matplotlib` from source. Set
`PYTHON=python3 ./run.sh` if your system is the other way around.

## Testing

```bash
pip install -r requirements-dev.txt
pytest                  # full suite, including the ~90s nested-CV regression test
pytest -m "not slow"    # fast subset only
```

## Environment

Tested on Python 3.12; `config.py` enforces a minimum of Python 3.9 at
import time. Dependency versions are pinned in `requirements.txt` — see
that file for exactly what was tested against.

## Notes for the write-up

- Random Forest was chosen for its robustness to noisy/mixed-type clinical
  data, resistance to overfitting relative to a single decision tree, and
  the interpretability afforded by feature importance scores.
- `class_weight="balanced"` is used in the classifier to reduce bias from
  the mild class imbalance (164 vs 139 in the raw multi-class counts).
- Raw Random Forest probabilities are not automatically trustworthy as
  probabilities, only as rankings — `output/calibration_curve.png` compares
  the raw pipeline's reliability diagram against the calibrated
  (`CalibratedClassifierCV`, sigmoid) one. For a health-related model, a
  calibrated probability is what makes "73% probability of disease"
  actually mean something, rather than just being a score that ranks
  patients relative to each other.
- Impurity importance (`feature_importance.png`) and permutation importance
  (`permutation_importance.png`) are both included and compared, since
  impurity importance is biased toward high-cardinality continuous
  features like `age` and `chol`. `ca`, `cp`, and `thal` dominate under
  both methods — consistent with established clinical risk factors, which
  strengthens the model's face validity.
