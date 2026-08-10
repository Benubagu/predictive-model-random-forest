# Data Dictionary

Source: Cleveland Heart Disease dataset (UCI Machine Learning Repository),
`data/heart.csv` — 303 patient records, 13 clinical attributes plus the
original multi-class diagnosis label.

| Attribute | Type | Description |
|---|---|---|
| age | numeric | Age in years |
| sex | binary | 1 = male, 0 = female |
| cp | categorical (1-4) | Chest pain type |
| trestbps | numeric | Resting blood pressure (mm Hg) |
| chol | numeric | Serum cholesterol (mg/dl) |
| fbs | binary | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| restecg | categorical (0-2) | Resting electrocardiographic results |
| thalach | numeric | Maximum heart rate achieved |
| exang | binary | Exercise-induced angina (1 = yes, 0 = no) |
| oldpeak | numeric | ST depression induced by exercise relative to rest |
| slope | categorical (1-3) | Slope of the peak exercise ST segment |
| ca | categorical (0-3) | Number of major vessels colored by fluoroscopy |
| thal | categorical (3, 6, 7) | Thalassemia: 3 = normal, 6 = fixed defect, 7 = reversible defect |
| diagnosis | categorical (0-4) | Original multi-class severity label, source column for the binary target |

## Target

`preprocessing.clean_data()` collapses `diagnosis` into a binary `target`
column: `0` = no disease, any value `> 0` = disease present.

## Known data quality notes

- A small number of missing values are encoded as `?` in the raw CSV
  (mainly in `ca` and `thal`). `preprocessing.py` converts these to `NaN`
  but does not impute — imputation happens inside the training pipeline
  (see `evaluation.build_pipeline`), refit on each CV fold, so the median
  used for a held-out patient never comes from that patient's own fold.
- Class balance in the raw multi-class counts is mildly imbalanced
  (164 vs 139), which is why `RandomForestClassifier` is trained with
  `class_weight="balanced"`.
- `data/external/` holds three sibling UCI cohorts (Hungarian, Switzerland,
  VA Long Beach) used for external validation only — see
  `external_validation.py` and the model card. They are **not** used for
  training. Two of them encode missing cholesterol as a literal `0`
  instead of `?` (all 123 Switzerland rows; 49/200 VA rows) — physiologically
  impossible in a living patient. `clean_data()` treats `chol`/`trestbps`
  values of exactly `0` as missing for this reason; Cleveland has no such
  rows, so this is a no-op on the training data.
