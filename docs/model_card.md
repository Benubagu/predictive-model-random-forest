# Model Card: Heart Disease Random Forest Classifier

## ⚠️ Not for clinical use

This model is a coursework / research artifact. It has not been
clinically validated, is not a medical device, has not been reviewed by
a regulatory body, and must not be used to make or influence real
diagnostic or treatment decisions. It exists to demonstrate a rigorous
ML methodology on a small, well-known public dataset — not to diagnose
anyone.

## Model details

- **Type**: Random Forest classifier (scikit-learn), wrapped in a
  `SimpleImputer` (median) → `RandomForestClassifier` pipeline, then
  wrapped again in `CalibratedClassifierCV` (sigmoid/Platt scaling) for
  the deployed model.
- **Final hyperparameters** (retuned on the full training set):
  `max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100`,
  `class_weight="balanced"`.
- **Decision threshold**: 0.164 (not 0.5) on calibrated probabilities,
  chosen to guarantee ≥95% sensitivity — see "Operating point" below.
- **Training procedure**: `src/training/train_model.py`. Methodology:
  `src/core/evaluation.py` (nested cross-validation) — see `README.md`
  for the full rationale.

## Intended use

Educational / research demonstration of a correctly-specified ML
pipeline (leakage-free preprocessing, nested CV, calibration, threshold
selection informed by clinical cost asymmetry, permutation importance,
external validation) applied to a small clinical dataset. Appropriate
uses: coursework, methodology demonstration, a starting point for
further research. **Not appropriate**: any use that influences an actual
patient's care.

## Training data

Cleveland Heart Disease dataset (UCI Machine Learning Repository),
`data/heart.csv` — 303 patients, 13 clinical attributes, collected at
the Cleveland Clinic Foundation. See `docs/data_dictionary.md` for the
full attribute reference and known data quality notes. Binary target:
disease present (any severity 1-4) vs. absent.

Class balance: 164 no-disease / 139 disease (46% prevalence).

## Evaluation

### In-distribution: nested cross-validation on Cleveland

A single 80/20 train/test split on 303 patients was judged too noisy to
report confidently (see README for why). These are 5-outer/5-inner
nested CV results: mean ± std across the 5 outer folds, and — where
noted — 95% bootstrap CIs on the pooled out-of-fold predictions.

| Metric | Mean ± std | 95% CI |
|---|---|---|
| Accuracy | 83.5% ± 3.9% | [78.9%, 87.5%] |
| Precision | 84.1% ± 4.7% | [77.7%, 90.2%] |
| Recall (sensitivity) | 79.1% ± 7.0% | [71.8%, 85.5%] |
| F1 score | 81.4% ± 4.7% | [76.0%, 86.0%] |
| ROC-AUC | 0.908 ± 0.027 | [0.874, 0.938] |
| PR-AUC | 0.902 ± 0.037 | [0.856, 0.938] |

These are threshold=0.5, raw (uncalibrated) pipeline numbers — the
metrics a reader would expect from a standard classification report.
This accuracy figure (not the operating-threshold one below) is what
the thesis's H1₁ hypothesis is tested against.

### Hypothesis tests

Computed live by `src.core.evaluation.evaluate_hypotheses` from the numbers
above and saved to `output/metrics.json` → `hypothesis_tests` (not
hand-typed, so it cannot silently drift from the code). Full discussion:
`docs/report.md` § "Hypothesis testing".

- **H1₁** (accuracy ≥ 80%): supported on the point estimate (83.5%) but
  the 95% CI lower bound (78.9%) falls just under 80% — reported as a
  marginal, not clean, pass.
- **H1₂** (attributes differ in contribution): decisively supported —
  `ca`, `cp`, `thal` exceed one std from zero permutation importance; the
  other ten attributes do not.

### Operating point (deployed threshold)

For deployment, the threshold is tuned on *calibrated* probabilities to
guarantee ≥95% sensitivity, since a missed disease case is far costlier
than a false alarm in a screening context:

| | Value |
|---|---|
| Threshold | 0.164 |
| Sensitivity | 97.1% |
| Specificity | 51.8% |

This is a deliberate trade-off, not an oversight: the model over-refers
borderline patients rather than risk missing disease. A different
`--target-sensitivity` would move this trade-off; see
`python -m src.training.train_model --help`.

### Out-of-distribution: external validation

Not required by the study's stated scope — thesis §1.6 lists external
validation among directions for *future* studies — but included here to
characterise generalisation honestly rather than leaving it for later.
The model — trained, imputed, and calibrated exclusively on Cleveland
data — was scored as-is against three sibling UCI cohorts it never saw,
collected at different sites with different equipment and data
practices (`src/analysis/external_validation.py`):

| Cohort | n | Prevalence | ROC-AUC | Accuracy @ 0.5 | Sensitivity @ operating threshold | Specificity @ operating threshold |
|---|---|---|---|---|---|---|
| Hungarian | 294 | 36% | 0.894 | 82.7% | 92.5% | 65.4% |
| Switzerland | 123 | 94% | 0.773 | 57.7% | 93.9% | 37.5% |
| VA Long Beach | 200 | 75% | 0.741 | 62.0% | 90.6% | 21.6% |
| All three combined | 617 | 60% | 0.861 | 71.0% | 92.2% | 55.5% |

**Reading this honestly**: the model generalizes reasonably to
Hungarian (ROC-AUC 0.894, close to Cleveland's 0.908) but degrades on
Switzerland and VA Long Beach. Two confounds, both real and both
documented rather than papered over:

1. **Case-mix shift** — Switzerland (94%) and VA (75%) have far higher
   disease prevalence than Cleveland (46%), so accuracy-at-0.5 is not
   comparable across cohorts; ROC-AUC (threshold-independent) is the
   fairer comparison, and even that drops meaningfully.
2. **Missingness and data quality** — Switzerland and VA have
   substantially more missing/miscoded data than Cleveland (see
   `docs/data_dictionary.md`; the cholesterol miscoding-as-zero issue in
   particular affects nearly all Switzerland rows), and the model's
   imputer only knows the Cleveland training distribution.

Sensitivity holds up reasonably well across all cohorts (90–94%) even
where specificity collapses — consistent with a threshold explicitly
tuned to prioritize catching disease over avoiding false alarms, though
a >90% loss in specificity at VA is a real generalization failure, not
a minor caveat.

### Baseline comparison (out of scope — see appendix)

A supplementary comparison against Logistic Regression and a Decision
Tree is available in `docs/appendix_baseline_comparison.md`. It is kept
out of this document's main results because thesis §1.7 (Scope of the
Study) states the study is "experimentally limited to the Random Forest
algorithm" and that other algorithms "are considered in the literature
review but are not experimentally implemented or compared as part of
this study" — the comparison exists and is documented, but is not a
finding of this thesis.

## Feature importance

Impurity importance (biased toward high-cardinality continuous
features) and permutation importance (computed on genuinely held-out
folds) agree on `ca`, `cp`, and `thal` as the strongest predictors —
consistent with established clinical risk factors. They disagree on
`age`, `chol`, and `thalach`, which rank highly under impurity
importance but not under permutation importance; see
`output/feature_importance.png` and `output/permutation_importance.png`.

## Limitations

- n=303 training patients is small; confidence intervals above are
  correspondingly wide.
- Single-site training data (Cleveland Clinic, one hospital, one
  demographic/referral population, data collected decades ago) —
  external validation shows this doesn't transfer uniformly.
- No re-calibration or re-tuning was performed for the external cohorts;
  they are scored exactly as the Cleveland-trained model would score a
  new incoming patient, which is the point of the experiment but also
  means specificity at VA/Switzerland is genuinely poor.
- 13 attributes only; no imaging, labs beyond what's listed, or
  longitudinal data.

## Recommendations for anyone extending this work

- Treat the external validation numbers, not the Cleveland-only numbers,
  as the honest estimate of how this would perform on a genuinely new
  patient population.
- If pursuing this further, see `docs/appendix_baseline_comparison.md`:
  Logistic Regression's competitiveness there is worth taking seriously
  in a future study whose scope includes a multi-algorithm comparison.
- Re-run `python -m src.training.train_model --target-sensitivity` at a
  different value before assuming 0.95 is the right clinical target — that number was a
  reasonable default choice for this exercise, not a validated clinical
  guideline.
