# Predictive Model for Heart Disease Diagnosis Using Random Forest Algorithm

A complete, working machine learning system that diagnoses the likely presence
of heart disease from patient clinical data using the Random Forest algorithm.

> **Not for clinical use.** This model is a research/coursework artifact. It
> has not been clinically validated and its predictions are not intended to
> replace professional clinical judgement or established diagnostic
> procedures — full disclaimer and limitations in `docs/model_card.md`.

For a narrative walkthrough (EDA → preprocessing → methodology → results),
see `docs/report.md`; for exact metrics, confidence intervals, and
limitations, see `docs/model_card.md`.

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
├── app.py                          # Streamlit GUI (entry point — run from repo root)
├── README.md
├── src/                             # All library/application code
│   ├── core/                        # Shared foundations, imported by every entry point
│   │   ├── config.py                 # Paths, seeds, CV folds, param grid
│   │   ├── preprocessing.py          # Data loading, cleaning, integrity check
│   │   └── evaluation.py             # Nested CV, bootstrap CIs, threshold selection, hypothesis tests
│   ├── training/
│   │   └── train_model.py            # Orchestrates training + evaluation, saves artifacts
│   ├── inference/
│   │   └── predict.py                # CLI diagnosis tool + predict_patient() used by app.py
│   └── analysis/                     # Optional, supplementary analyses
│       ├── external_validation.py     # Scores the trained model against external cohorts
│       ├── benchmark.py               # Random Forest vs. Logistic Regression vs. Decision Tree
│       └── tuning_effect.py           # Default vs. tuned Random Forest (RQ3)
├── tests/                           # pytest regression suite (mirrors src/ by import, not by folder)
├── data/
│   ├── heart.csv                     # Raw training dataset (Cleveland)
│   └── external/                     # Hungarian/Switzerland/VA cohorts — validation only, never trained on
├── docs/
│   ├── data_dictionary.md            # Attribute reference + data quality notes
│   ├── model_card.md                 # Intended use, metrics + CIs, limitations, disclaimer
│   ├── report.md                     # EDA -> preprocessing -> methodology -> results write-up
│   ├── thesis_traceability.md        # Maps thesis objectives/RQs/hypotheses to code + artifacts
│   ├── literature_comparison.md      # RQ5: comparison against prior published studies
│   └── appendix_baseline_comparison.md
├── model/                          # Trained pipeline + threshold (created by train_model.py, gitignored)
├── output/                         # Metrics + plots (created by train_model.py etc., gitignored)
├── run.sh                          # One command: clean + train (or install/test/benchmark/... alone)
├── pytest.ini
├── requirements.txt                # Core dependencies (training/inference)
├── requirements-dev.txt            # requirements.txt + pytest
├── requirements-app.txt            # requirements.txt + streamlit (for app.py only)
├── .gitignore
└── .gitattributes
```

**Why this layout.** `src/` holds everything importable; `app.py` and
`README.md` are the only code-adjacent files left at the repository root,
because they're the two things meant to be pointed at directly (`streamlit
run app.py`, and the doc you're reading). Inside `src/`, modules are grouped
by role rather than dumped flat: `core/` has no dependencies on the rest of
the project and is imported by everything else; `training/`, `inference/`,
and `analysis/` each depend on `core/` but not on each other. `tests/` stays
a sibling of `src/` rather than nested inside it — this is the conventional
Python "src layout" (tests treat the package as an external consumer,
importing it the same way a user would).

## How it works

1. **`src/core/preprocessing.py`** — loads the raw CSV, converts `?`
   missing-value markers to NaN, casts columns to numeric, and derives the
   binary diagnosis label. It does **not** impute missing values: imputation
   is a fold-dependent statistic, so it lives inside the sklearn `Pipeline`
   (`SimpleImputer` → `RandomForestClassifier`) built in
   `src/core/evaluation.py` and is refit on every training fold. Nothing
   about a held-out patient, in any fold, influences their own prediction.
2. **`src/core/evaluation.py`** — runs nested cross-validation: an outer
   loop (5 folds) produces the honest performance estimate, while
   hyperparameter tuning happens on an independent inner loop (5 folds)
   inside each outer training fold. The outer loop also produces
   leakage-free out-of-fold probability predictions for the entire dataset
   — both raw and calibrated (`CalibratedClassifierCV`, Platt/sigmoid) —
   plus per-fold permutation importance and the formal hypothesis-test
   evaluation.
3. **`src/training/train_model.py`** — orchestrates the above, bootstraps
   95% confidence intervals from the pooled out-of-fold predictions, picks
   a decision threshold that guarantees a minimum sensitivity (see below),
   refits a final calibrated pipeline on the full dataset for deployment,
   and saves the model, threshold, metrics, and plots.
4. **`src/inference/predict.py`** — loads the saved pipeline and the tuned
   threshold, then diagnoses a new patient (missing fields are left blank
   and imputed by the pipeline) either interactively via the CLI or
   programmatically via `predict_patient()`.
5. **`app.py`** — a Streamlit GUI wrapping `predict_patient()` unchanged
   (see "Running the Streamlit app" below).
6. **`src/analysis/external_validation.py`** *(optional, run after
   training)* — scores the trained model against three external UCI
   cohorts it never saw.
7. **`src/analysis/benchmark.py`** *(optional, supplementary)* — runs the
   same nested CV protocol over Random Forest, Logistic Regression, and a
   Decision Tree for an apples-to-apples comparison.
8. **`src/analysis/tuning_effect.py`** *(optional)* — compares
   default-hyperparameter vs. tuned Random Forest under the same protocol.

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

### Does it generalize? External validation

Not required by the thesis's stated scope (§1.6 lists it as future work)
but included to characterise generalisation honestly rather than leaving
it for later. `src/analysis/external_validation.py` scores the
Cleveland-trained model — no retraining, no re-imputation — against three
sibling UCI cohorts it never saw:

| Cohort | n | ROC-AUC |
|---|---|---|
| Hungarian | 294 | 0.894 |
| Switzerland | 123 | 0.773 |
| VA Long Beach | 200 | 0.741 |

Holds up well on Hungarian; degrades on Switzerland/VA, where both disease
prevalence and missingness differ substantially from Cleveland. Full
numbers, the confounds, and what this does and doesn't imply about
real-world robustness: `docs/model_card.md`.

### Supplementary: baseline comparison (out of thesis scope)

`src/analysis/benchmark.py` compares Random Forest against Logistic
Regression and a Decision Tree under an identical nested CV protocol.
This is **not** a finding of the thesis — §1.7 (Scope of the Study)
states the study is "experimentally limited to the Random Forest
algorithm" and that other algorithms "are considered in the literature
review but are not experimentally implemented or compared as part of
this study." The comparison is implemented and preserved, not deleted,
but documented separately: see `docs/appendix_baseline_comparison.md`.

---

## Requirements

| | Requirement |
|---|---|
| **Python** | 3.9+ (developed and tested on 3.12). Enforced at import time by `src/core/config.py`. |
| **OS** | Cross-platform (Windows, macOS, Linux). `run.sh` requires a POSIX shell — Git Bash/WSL on Windows, or the native shell on macOS/Linux. |
| **Package manager** | `pip` (no `conda`/`poetry` requirement — plain `venv` + `pip` is enough). |
| **Hardware** | CPU only. No GPU needed. A full training run (grid search + nested CV) takes roughly 4–5 minutes on a typical laptop CPU. |
| **Disk** | A few hundred MB for dependencies; the dataset and generated artifacts are a few MB. |
| **Network** | Only needed once, to `pip install` dependencies (and, optionally, to re-download the external-validation CSVs, which are already committed under `data/external/`). |

Dependencies are split into three tiers so you only install what you need:

| File | Installs | Needed for |
|---|---|---|
| `requirements.txt` | pandas, numpy, scikit-learn, matplotlib, joblib | Training, inference, all `src/analysis/*` scripts |
| `requirements-dev.txt` | `requirements.txt` + pytest | Running the test suite |
| `requirements-app.txt` | `requirements.txt` + streamlit | Running the Streamlit GUI (`app.py`) |

All versions are pinned to what this repository was actually developed and
tested against — see each file for exact numbers.

## Step-by-step: getting started from a clean clone

```bash
# 1. Clone and enter the repository
git clone <this-repo-url>
cd predictive-model-random-forest

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows (cmd/PowerShell)

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Train the model (~4-5 minutes; regenerates model/ and output/ from data/heart.csv)
python -m src.training.train_model

# 5. Try a prediction interactively
python -m src.inference.predict
```

That's the whole path from clone to a working, trained model. Everything
below is optional, in roughly the order you'd reach for it next.

```bash
# Or, steps 4 in one command (clean + train):
./run.sh

# Run the test suite (needs requirements-dev.txt)
pip install -r requirements-dev.txt
python -m pytest

# Launch the Streamlit GUI instead of the CLI (needs requirements-app.txt)
pip install -r requirements-app.txt
streamlit run app.py

# Optional supplementary analyses (each independent, each ~seconds-to-minutes)
python -m src.analysis.external_validation   # score against external UCI cohorts
python -m src.analysis.benchmark              # RF vs. Logistic Regression vs. Decision Tree
python -m src.analysis.tuning_effect          # default vs. tuned RF (RQ3)
```

**Run everything from the repository root.** The `-m src.____.____` form is
required (not `python src/training/train_model.py` directly) because the
internal imports (e.g. `from src.core import config`) are absolute imports
rooted at the repo root — `-m` puts the repo root on Python's import path;
direct file invocation does not. `run.sh` already `cd`s to the repo root
for you, so `./run.sh <target>` always works regardless of where you call
it from.

### Running the Streamlit app

```bash
pip install -r requirements-app.txt
streamlit run app.py
```

Opens a browser form with all 13 clinical attributes (dropdowns for
categorical fields, number inputs for continuous ones, each with an
"unknown" option that imputes the same way the CLI does). Click **Predict**
to see the label, the calibrated probability, and the decision threshold
in use. The disclaimer banner is always visible on screen. `app.py` itself
contains no prediction logic — it calls the exact same `load_model`,
`load_threshold`, and `predict_patient` functions as the CLI
(`src/inference/predict.py`), verified in `tests/test_app.py` by identity,
not just by convention.

### `run.sh` reference

```bash
./run.sh            # clean + train (default)
./run.sh install     # pip install -r requirements.txt
./run.sh train       # train only, skip clean
./run.sh test        # pytest, fast subset (skips the @slow nested-CV regression test)
./run.sh clean       # remove generated model/output artifacts
./run.sh external    # score the trained model against external UCI cohorts
./run.sh benchmark   # supplementary baseline comparison (see appendix)
./run.sh tuning      # RQ3: default vs. tuned Random Forest
```

It prefers the `python` command over `python3` — on Windows, `python3` can
resolve to the App Execution Alias stub under `WindowsApps/`, a distinct,
partially-populated Python environment that will fail trying to build
`matplotlib` from source. Set `PYTHON=python3 ./run.sh` if your system is
the other way around.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest                  # full suite, including the ~90s nested-CV regression test
python -m pytest -m "not slow"    # fast subset only
```

`pytest.ini` sets `pythonpath = .`, so `src` (and `app.py`) are importable
by the test suite regardless of how pytest is invoked — bare `pytest` also
works, not just `python -m pytest`.

## Environment

Tested on Python 3.12; `src/core/config.py` enforces a minimum of Python
3.9 at import time. Dependency versions are pinned in `requirements.txt`
— see that file for exactly what was tested against.

## Notes for the write-up

- Random Forest was chosen for its robustness to noisy/mixed-type clinical
  data, resistance to overfitting relative to a single decision tree, and
  the interpretability afforded by feature importance scores — see
  `docs/appendix_baseline_comparison.md` for a supplementary (out-of-scope)
  comparison against simpler models.
- `class_weight="balanced"` is used in the classifier to reduce bias from
  the mild class imbalance (164 vs 139 in the raw multi-class counts).
- Raw Random Forest probabilities are not automatically trustworthy as
  probabilities, only as rankings — `output/calibration_curve.png` compares
  the raw pipeline's reliability diagram against the calibrated
  (`CalibratedClassifierCV`, sigmoid) one. For a health-related model, a
  calibrated probability is what makes "80% probability of disease"
  actually mean something, rather than just being a score that ranks
  patients relative to each other.
- Impurity importance (`feature_importance.png`) and permutation importance
  (`permutation_importance.png`) are both included and compared, since
  impurity importance is biased toward high-cardinality continuous
  features like `age` and `chol`. `ca`, `cp`, and `thal` dominate under
  both methods — consistent with established clinical risk factors, which
  strengthens the model's face validity.
