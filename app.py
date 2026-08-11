"""
app.py
-------
Streamlit GUI for the trained Random Forest heart disease model
(Objective 6 / RQ6: "a functional prediction interface"). This is a thin
UI layer only -- all prediction logic is imported unchanged from
src/inference/predict.py (load_model, load_threshold, predict_patient),
so the CLI and this app can never diverge in behavior.

This file must stay at the repository root (not under src/): it's the
one entry point meant to be launched directly by name (`streamlit run
app.py`), so it needs to sit where that command is naturally run from.

Usage (from the repository root):
    streamlit run app.py
"""

import numpy as np
import streamlit as st

from src.inference.predict import load_model, load_threshold, predict_patient
from src.core.preprocessing import FEATURE_DESCRIPTIONS, FEATURE_NAMES, VALID_VALUES

UNKNOWN = "Unknown"

# Reasonable default values for the form, purely for a non-empty initial
# screen -- not derived from or influencing the model in any way.
DEFAULTS = {
    "age": 54.0, "sex": 1, "cp": 4, "trestbps": 130.0, "chol": 240.0,
    "fbs": 0, "restecg": 0, "thalach": 150.0, "exang": 0, "oldpeak": 1.0,
    "slope": 2, "ca": 0, "thal": 3,
}

CONTINUOUS_FEATURES = [f for f in FEATURE_NAMES if f not in VALID_VALUES]
CATEGORICAL_FEATURES = [f for f in FEATURE_NAMES if f in VALID_VALUES]


@st.cache_resource
def get_model_and_threshold():
    return load_model(), load_threshold()


def render_inputs():
    patient = {}
    col1, col2 = st.columns(2)
    columns = [col1, col2]

    for i, feat in enumerate(FEATURE_NAMES):
        col = columns[i % 2]
        desc = FEATURE_DESCRIPTIONS[feat]
        with col:
            if feat in CATEGORICAL_FEATURES:
                options = [UNKNOWN] + sorted(VALID_VALUES[feat])
                default_index = options.index(DEFAULTS[feat]) if DEFAULTS[feat] in options else 0
                choice = st.selectbox(f"{feat} — {desc}", options, index=default_index, key=feat)
                patient[feat] = np.nan if choice == UNKNOWN else float(choice)
            else:
                unknown = st.checkbox(f"{feat} unknown", key=f"{feat}_unknown")
                value = st.number_input(
                    f"{feat} — {desc}", value=float(DEFAULTS[feat]), key=feat, disabled=unknown,
                )
                patient[feat] = np.nan if unknown else float(value)
    return patient


def main():
    st.set_page_config(page_title="Heart Disease Prediction (Research Demo)", page_icon="❤️")
    st.title("Heart Disease Prediction — Research Demo")

    st.warning(
        "**Not for clinical use.** This is a coursework / research artifact "
        "(Random Forest trained on the UCI Cleveland Heart Disease dataset, "
        "n=303). It has not been clinically validated and must not be used "
        "to make or influence real diagnostic or treatment decisions. See "
        "`docs/model_card.md` for full details and limitations."
    )

    st.write(
        "Enter patient clinical attributes below. Leave a field as "
        "**Unknown** (categorical) or check its **unknown** box "
        "(numeric) if not available — the trained pipeline imputes "
        "missing values the same way it did during training."
    )

    model, threshold = get_model_and_threshold()
    patient = render_inputs()

    if st.button("Predict", type="primary"):
        label, proba = predict_patient(model, patient, threshold)
        if label == "Heart Disease Likely":
            st.error(f"**Prediction: {label}**")
        else:
            st.success(f"**Prediction: {label}**")
        st.metric("Predicted probability of heart disease", f"{proba:.1%}")
        st.caption(
            f"Decision threshold: {threshold:.1%} (tuned for high sensitivity, "
            "not the default 0.5 — see docs/model_card.md § 'Operating point')."
        )


if __name__ == "__main__":
    main()
