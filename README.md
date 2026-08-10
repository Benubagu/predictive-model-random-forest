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
├── model/                          # Trained model (created by train_model.py, gitignored)
├── output/                         # Metrics + plots (created by train_model.py, gitignored)
├── preprocessing.py                # Data loading, cleaning, imputation
├── train_model.py                  # Grid search, training, evaluation
├── predict.py                      # CLI diagnosis tool for new patients
├── requirements.txt
├── .gitignore
└── README.md
```

## How it works

1. **preprocessing.py** — loads the raw CSV, converts `?` missing-value
   markers to NaN, imputes with the column median, and derives the binary
   diagnosis label.
2. **train_model.py** — splits the data 80/20 (stratified), runs a grid
   search over Random Forest hyperparameters (`n_estimators`, `max_depth`,
   `min_samples_split`, `min_samples_leaf`) with 5-fold cross-validation
   optimizing F1 score, evaluates the best model on the held-out test set,
   and saves the model plus evaluation plots.
3. **predict.py** — loads the saved model and diagnoses a new patient from
   their 13 clinical attributes, either interactively via the command line
   or programmatically via `predict_patient()`.

## Results obtained

- Accuracy: **88.5%**
- Precision: **83.9%**
- Recall: **92.9%**
- F1 score: **88.1%**
- ROC-AUC: **0.955**

(See `output/metrics.json` for exact figures and best hyperparameters, and
the PNG files for the confusion matrix, ROC curve, and feature importance
chart.)

## Running it

```bash
pip install -r requirements.txt

# Train (also re-generates the model + all output artifacts)
python3 train_model.py

# Diagnose a new patient interactively
python3 predict.py
```

## Notes for the write-up

- Random Forest was chosen for its robustness to noisy/mixed-type clinical
  data, resistance to overfitting relative to a single decision tree, and
  the interpretability afforded by feature importance scores.
- `class_weight="balanced"` is used in the classifier to reduce bias from
  the mild class imbalance (164 vs 139 in the raw multi-class counts).
- The feature importance chart is a useful discussion point: attributes
  like chest pain type, thalassemia, number of major vessels, and maximum
  heart rate typically dominate — consistent with established clinical
  risk factors, which strengthens the model's face validity.
