import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

model = joblib.load(ROOT / "models" / "lr_baseline_pipeline.joblib")

with open(ROOT / "models" / "risk_config.json", encoding="utf-8") as f:
    config = json.load(f)

st.set_page_config(
    page_title="UPI Fraud Risk Detection",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Real-Time UPI Fraud Risk Detection")
st.caption("Hackathon proof of concept — simulated decisions only")

# Get feature names and category values from fitted pipeline
feature_names = list(model.feature_names_in_)
# Read categorical feature names directly from ColumnTransformer
preprocessor = model.named_steps["preprocessing"]

category_cols = [
    cols
    for name, transformer, cols in preprocessor.transformers_
    if name == "cat"
][0]

encoder = (
    preprocessor
    .named_transformers_["cat"]
    .named_steps["encoder"]
)

category_options = {
    col: [str(v) for v in values]
    for col, values in zip(category_cols, encoder.categories_)
}

encoder = (
    model.named_steps["preprocessing"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
)

category_options = {
    col: [str(v) for v in values]
    for col, values in zip(category_cols, encoder.categories_)
}

with st.form("fraud_form"):
    amount = st.number_input(
        "Transaction Amount (₹)", min_value=0.0, value=1500.0
    )
    hour = st.slider("Hour of Day", 0, 23, 14)
    month = st.slider("Month", 1, 12, 10)
    weekend = st.checkbox("Weekend Transaction")

    transaction = {
        "amount (INR)": amount,
        "hour_of_day": hour,
        "month": month,
        "is_weekend": int(weekend)
    }

    for col in category_cols:
        transaction[col] = st.selectbox(
            col.replace("_", " ").title(),
            category_options[col]
        )

    submitted = st.form_submit_button("Analyze Transaction")

if submitted:
    input_df = pd.DataFrame([transaction])[feature_names]
    score = float(model.predict_proba(input_df)[0, 1])

    if score >= config["block_threshold"]:
        decision = "BLOCK"
    elif score >= config["otp_threshold"]:
        decision = "OTP"
    else:
        decision = "APPROVE"

    st.metric("Model Fraud Score", f"{score:.6f}")
    st.metric("Simulated Risk Decision", decision)

    if decision == "BLOCK":
        st.error("High-risk flag — simulated block, not an actual payment action.")
    elif decision == "OTP":
        st.warning("Additional verification suggested — simulation only.")
    else:
        st.success("Low model risk score — simulated approval.")

    st.caption(
        "This model has not demonstrated reliable fraud detection. "
        "Its scores are not calibrated probabilities, and the decisions "
        "must not be used for real financial transactions."
    )