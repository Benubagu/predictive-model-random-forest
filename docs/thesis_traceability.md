# Thesis Traceability Matrix

Maps every objective, research question, and hypothesis in
`docs/Heart_Disease_RF_1-2.docx` (Chapter One) to the specific code
artifact and output file that satisfies it. Status is read from an
actual completed run of this repository, not asserted.

Note on counting: the work order that requested this document described
"6 objectives, 5 research questions, and 2 hypotheses." The thesis text
itself (§1.3–§1.5) lists **6 objectives, 6 research questions (RQ1–RQ6),
and 2 hypotheses** (each as an H0/H1 pair). This table uses the thesis
text as authoritative and covers all 6 RQs.

## Objectives (§1.3)

| # | Objective (thesis wording) | Code | Output artifact | Status |
|---|---|---|---|---|
| 1 | Acquire and preprocess a clinical dataset suitable for heart disease prediction | `preprocessing.py::load_raw_data`, `clean_data`, `validate_raw_data` | `data/heart.csv`, `docs/data_dictionary.md` | ✅ Done |
| 2 | Develop and implement a Random Forest-based classification model | `evaluation.py::build_pipeline` | `model/random_forest_heart_model.joblib` | ✅ Done |
| 3 | Optimise the Random Forest model through hyperparameter tuning using cross-validated grid search | `evaluation.py::nested_cv_evaluate` (inner `GridSearchCV`), `config.PARAM_GRID` | `output/metrics.json` → `final_model_params`; RQ3 effect size: `tuning_effect.py` → `output/tuning_effect.json` | ✅ Done |
| 4 | Evaluate predictive performance using accuracy, precision, recall, F1, ROC-AUC | `evaluation.py::compute_metrics`, `nested_cv_evaluate` | `output/metrics.json` → `nested_cv_summary`, `bootstrap_ci`; `output/roc_curve.png`, `output/pr_curve.png` | ✅ Done |
| 5 | Identify clinical attributes contributing most to predictions via feature importance | `train_model.py` (impurity), `evaluation.py::nested_cv_evaluate` (permutation, per outer fold) | `output/feature_importance.png`, `output/permutation_importance.png`, `output/metrics.json` → `impurity_importance` / `permutation_importance_mean` | ✅ Done |
| 6 | Implement the trained model in a functional prediction interface | `predict.py` (CLI), `app.py` (Streamlit GUI) | Run `python predict.py` or `streamlit run app.py` | ✅ Done |

## Research Questions (§1.4)

| # | Research Question (thesis wording) | Code | Output artifact | Status |
|---|---|---|---|---|
| RQ1 | What level of predictive performance can a Random Forest model achieve? | `evaluation.py::nested_cv_evaluate` | `output/metrics.json` → `nested_cv_summary` (83.5% ± 3.9% accuracy, 95% CI [78.9%, 87.5%]) | ✅ Done |
| RQ2 | Which clinical attributes contribute most to the model's predictions? | `evaluation.py::nested_cv_evaluate` (permutation importance) | `output/permutation_importance.png`; `ca`, `cp`, `thal` lead | ✅ Done |
| RQ3 | To what extent does hyperparameter tuning affect predictive performance? | `tuning_effect.py` (default vs. tuned RF, identical nested CV protocol) | `output/tuning_effect.json`, `output/tuning_effect.png`; `docs/report.md` §4.1 | ✅ Done (was a gap; closed) |
| RQ4 | How does the model perform on accuracy, precision, recall, F1, ROC-AUC? | `evaluation.py::compute_metrics` | `output/metrics.json` → `nested_cv_summary`; `output/classification_report.txt` | ✅ Done |
| RQ5 | How does performance compare with prior studies on the Cleveland dataset? | `docs/literature_comparison.md` | Comparison table vs. Singh et al. (2017), Gavhane et al. (2018), Katarya & Srinivas (2020) | ✅ Done (was a gap; closed) |
| RQ6 | Can the trained model be implemented in a functional prediction interface? | `predict.py`, `app.py` | CLI + Streamlit GUI, both calling the same `predict_patient()` | ✅ Done |

## Hypotheses (§1.5)

| # | Hypothesis (thesis wording) | Code | Output artifact | Status |
|---|---|---|---|---|
| H0₁ / H1₁ | Tuned RF does/does not achieve ≥80% accuracy | `evaluation.py::evaluate_hypotheses` | `output/metrics.json` → `hypothesis_tests.h1_accuracy_at_least_target` | ⚠️ **Supported on the point estimate (83.5% ≥ 80%), not strictly at the 95% CI lower bound (78.9% < 80%)** — see `docs/report.md` §4.2 |
| H0₂ / H1₂ | Clinical attributes do/do not differ in relative contribution | `evaluation.py::evaluate_hypotheses` | `output/metrics.json` → `hypothesis_tests.h2_features_differ_in_contribution` | ✅ **Decisively supported** — `ca`, `cp`, `thal` exceed 1 std from zero; 10 other attributes do not |

## Scope items not required by the thesis, included anyway (see DECISION note)

| Item | Thesis status | Code | Framing |
|---|---|---|---|
| External validation (Hungarian/Switzerland/VA cohorts) | Listed as **future work** in §1.6 | `external_validation.py` | Included as "future work brought forward," honest degradation reported — see `docs/model_card.md` |
| Logistic Regression / Decision Tree comparison | Explicitly **out of scope** per §1.7 ("experimentally limited to the Random Forest algorithm... not experimentally implemented or compared") | `benchmark.py` | Quarantined to `docs/appendix_baseline_comparison.md`, not reported as a thesis finding — **decision required from the author, see commit history** |

## How to regenerate this table's numbers

```bash
python train_model.py        # nested CV, hypothesis tests, feature importance
python tuning_effect.py      # RQ3
python external_validation.py
python benchmark.py          # supplementary only, see appendix
```

All figures above are current as of the run these files were generated
from. If you re-run with a different seed, dataset, or config, re-check
this table against the fresh `output/*.json` files rather than trusting
the numbers here — they are a snapshot, not a live view.
