# Report: Heart Disease Prediction with Random Forest

A walkthrough of the dataset, methodology, and results in this repo —
written to stand alone as a write-up (dissertation appendix, coursework
submission) rather than as API documentation. For the "how do I run
this" version, see the main `README.md`; for exact reproducibility
figures with confidence intervals, see `docs/model_card.md`.

## 1. Data

The Cleveland Heart Disease dataset (UCI Machine Learning Repository,
`data/heart.csv`): 303 patients, 13 clinical attributes, collected at
the Cleveland Clinic Foundation. Full attribute reference:
`docs/data_dictionary.md`.

**Class balance**: 164 no-disease / 139 disease (46% prevalence) — mild
imbalance, handled with `class_weight="balanced"` rather than
resampling.

**Demographics**: ages 29–77 (mean 54.4); 206 male / 97 female (68%/32%
— a known skew in this dataset, not corrected for here, and worth
flagging as an external-validity caveat rather than silently ignoring).

**Missingness**: minimal in Cleveland — 4 missing `ca` values, 2 missing
`thal`, nothing else. This is why a naive single-impute-then-split
pipeline "worked" well enough to not obviously break in an earlier
version of this project (see `preprocessing.py`'s docstring) — the
leakage was real but small, because there was almost nothing to leak.
That stops being true for the external cohorts (§5), where missingness
is substantial.

**Linear correlation with the target** (Pearson, sorted by magnitude) —
a first, crude look at which features carry signal before any modeling:

| Feature | Correlation |
|---|---|
| thal | 0.526 |
| ca | 0.460 |
| exang | 0.432 |
| oldpeak | 0.425 |
| thalach | -0.417 |
| cp | 0.414 |
| slope | 0.339 |
| sex | 0.277 |
| age | 0.223 |
| restecg | 0.169 |
| trestbps | 0.151 |
| chol | 0.085 |
| fbs | 0.025 |

`fbs` and `chol` show almost no linear relationship with the target —
worth keeping in mind going into §4, where permutation importance
demotes both.

## 2. Preprocessing

`preprocessing.py` converts `?` missing-value markers to `NaN`, casts
columns to numeric, and derives the binary target
(`diagnosis > 0`). It deliberately does **not** impute — see §3.

One data-quality fix lives here that matters more for external cohorts
than for Cleveland: `chol`/`trestbps` values of exactly `0` are treated
as missing. Cleveland has none of these (a value of 0 for either is
physiologically impossible), but two of the external cohorts encode
missing cholesterol this way — see §5.

## 3. Methodology

The methodology in this repo went through several iterations, each
fixing a specific correctness or rigor problem. The short version, in
order of how much they matter:

1. **Imputation moved inside the CV loop.** Originally, missing values
   were median-imputed once on the full dataset before splitting into
   train/test — a train/test leak, because the test set's own values
   contributed to the training-time statistic used to fill it in.
   `evaluation.build_pipeline()` now wraps `SimpleImputer` and the
   classifier in one `sklearn.Pipeline`, refit on every training fold.
2. **Nested cross-validation replaced a single 80/20 split.** A
   61-patient test set is too small to report accuracy to a tenth of a
   percent with any confidence. `evaluation.nested_cv_evaluate()` uses
   an outer loop (5 folds) for the performance estimate and an inner
   loop (5 folds, run independently per outer fold) for hyperparameter
   search, so every patient is a test case exactly once and no fold's
   hyperparameters were chosen using its own test data.
3. **Calibration.** A Random Forest's `predict_proba` output is a good
   *ranking* but not necessarily a trustworthy *probability*. The
   deployed model wraps the tuned pipeline in `CalibratedClassifierCV`
   (Platt/sigmoid scaling); `output/calibration_curve.png` shows the
   raw vs. calibrated reliability diagrams side by side.
4. **Threshold tuning.** A missed disease case (false negative) is far
   costlier than a false alarm in a screening context, so the decision
   threshold is chosen to guarantee ≥95% sensitivity rather than
   defaulting to 0.5 — see `evaluation.select_operating_threshold()`.
5. **Permutation importance alongside impurity importance**, since
   impurity importance is known to be biased toward high-cardinality
   continuous features (`age`, `chol`) — see §4.

## 4. Results

Full figures with confidence intervals are in `docs/model_card.md`;
summarized here.

Nested CV (mean ± std across 5 outer folds), raw pipeline at threshold
0.5:

| Metric | Value |
|---|---|
| Accuracy | 83.5% ± 3.9% |
| ROC-AUC | 0.908 ± 0.027 |
| PR-AUC | 0.902 ± 0.037 |

At the tuned operating threshold (0.164, calibrated probabilities):
97.1% sensitivity / 51.8% specificity — a deliberate over-referral
trade-off for a screening context.

**Feature importance** (`output/feature_importance.png` vs.
`output/permutation_importance.png`): both methods agree `ca`, `cp`,
and `thal` matter most, matching the linear-correlation table in §1 and
established clinical risk factors. They disagree on `age` and `chol`,
which impurity importance ranks highly but permutation importance (and
the raw correlation) does not — a textbook case of impurity importance
overweighting a high-cardinality continuous feature.

**Baseline comparison** (`benchmark.py`, `output/model_comparison.png`):
Logistic Regression is statistically indistinguishable from Random
Forest on every metric (83.8% vs. 83.5% accuracy, 0.909 vs. 0.908
ROC-AUC), with *lower* variance, and it's far more interpretable. A
single Decision Tree clearly underperforms both (78.5% accuracy, 0.792
ROC-AUC). The honest conclusion is that Random Forest's added
complexity does not buy performance on this dataset — see
`docs/model_card.md` for the full discussion.

## 5. External validation

The Cleveland-trained model was scored — no retraining, no
re-imputation, no re-calibration — against three sibling UCI cohorts
(Hungarian, Switzerland, VA Long Beach), collected at different sites:

| Cohort | n | Prevalence | ROC-AUC |
|---|---|---|---|
| Hungarian | 294 | 36% | 0.894 |
| Switzerland | 123 | 94% | 0.773 |
| VA Long Beach | 200 | 75% | 0.741 |

Hungarian — closest to Cleveland in prevalence and (per the UCI
documentation) collection practices — holds up well. Switzerland and VA
degrade substantially, driven by two compounding factors: very
different disease prevalence (case-mix shift) and much heavier
missingness, including cholesterol being miscoded as `0` for nearly all
Switzerland rows and a quarter of VA rows (§2). This is presented as a
genuine limitation, not explained away — see `docs/model_card.md` §
"Limitations" for the full discussion of what this does and doesn't
tell us about the model's real-world robustness.

## 6. What this report is not

This is a demonstration of methodology on a small public dataset, not a
validated clinical tool. See the disclaimer at the top of
`docs/model_card.md`.
