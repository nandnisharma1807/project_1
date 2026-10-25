"""Streamlit frontend — Diabetes Analytics & Prediction.

Run from the project root:
    streamlit run frontend/app.py

Pages (tabs):
  1. Overview & Data — cleaned data, class balance, distributions
  2. Model Comparison — every iteration, with accuracy / precision /
     recall / F1 / ROC-AUC, and the winner's confusion matrix
  3. Predict — single-patient form + batch CSV prediction

This is an analytics / screening-support tool, not a medical diagnosis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.predict import FEATURES, load_model, predict_batch, predict_single  # noqa: E402

st.set_page_config(page_title="Diabetes Analytics", layout="wide")
st.title("🩺 Diabetes Analytics & Prediction")
st.caption("Cleaned Pima Indians Diabetes data · multiple models compared · best model chosen by F1-score. "
           "For analytics and screening support only — not a medical diagnosis.")

# --- Load artefacts -------------------------------------------------------
try:
    model = load_model()
    model_loaded = True
except Exception as e:  # noqa: BLE001
    model, model_loaded = None, False
    st.error(f"Best model not found. Run `python backend/train.py` first. ({e})")

comparison_path = ROOT / "results" / "model_comparison.csv"
results_path = ROOT / "results" / "results.json"
clean_path = ROOT / "data" / "diabetes_clean.csv"
cm_path = ROOT / "results" / "confusion_matrix_best.png"

import json  # noqa: E402

results = json.loads(results_path.read_text()) if results_path.exists() else {}
best_name = results.get("best_model", "—")
best_metrics = results.get("best_metrics", {})

with st.sidebar:
    st.header("Best model")
    st.metric("Model", best_name)
    if best_metrics:
        st.metric("F1-score", f"{best_metrics.get('f1', 0):.4f}")
        st.metric("Recall", f"{best_metrics.get('recall', 0):.4f}")
        st.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.4f}")
        st.metric("Precision", f"{best_metrics.get('precision', 0):.4f}")
    st.divider()
    st.write("**Cleaning applied:** impossible zeros in Glucose, "
             "BloodPressure, SkinThickness, Insulin and BMI treated as "
             "missing and filled with the median. No rows dropped.")

tab_overview, tab_models, tab_predict = st.tabs(["Overview & Data", "Model Comparison", "Predict"])

# --- Tab 1: Overview ------------------------------------------------------
with tab_overview:
    st.subheader("Cleaned dataset")
    if clean_path.exists():
        df = pd.read_csv(clean_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(df))
        col2.metric("Features", len(FEATURES))
        col3.metric("Diabetes cases", int((df["Outcome"] == 1).sum()))
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("Outcome balance")
        st.bar_chart(df["Outcome"].value_counts().rename({0: "No diabetes", 1: "Diabetes"}))
        st.subheader("Feature distributions by outcome")
        feature = st.selectbox("Feature", FEATURES, index=1)
        st.bar_chart(df.groupby("Outcome")[feature].mean())
    else:
        st.warning("Cleaned data not found. Run `python backend/train.py` first.")

# --- Tab 2: Model comparison ----------------------------------------------
with tab_models:
    st.subheader("All iterations — same 80/20 stratified split (seed 42)")
    if comparison_path.exists():
        comp = pd.read_csv(comparison_path)
        st.dataframe(comp, use_container_width=True)
        st.bar_chart(comp.set_index("model")["f1"], height=300)
        st.write("**Winner (highest F1, then recall, then accuracy):**", best_name)
        if cm_path.exists():
            st.subheader(f"Confusion matrix — {best_name}")
            st.image(str(cm_path), width=520)
            cm = results.get("confusion_matrix_best", [[0, 0], [0, 0]])
            st.write(f"TN = {cm[0][0]}, FP = {cm[0][1]}, FN = {cm[1][0]}, TP = {cm[1][1]}. "
                     "Recall = TP / (TP + FN); a low FN count is what matters most for screening.")
        importance = results.get("feature_importance_or_coefficients")
        if importance:
            st.subheader("Feature importance / coefficients (best model)")
            st.bar_chart(pd.Series(importance))
    else:
        st.warning("Model comparison not found. Run `python backend/train.py` first.")

# --- Tab 3: Predict ---------------------------------------------------------
with tab_predict:
    st.subheader("Single-patient prediction")
    if not model_loaded:
        st.stop()
    cols = st.columns(4)
    values = {}
    defaults = {"Pregnancies": 1, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
                "Insulin": 80, "BMI": 28.0, "DiabetesPedigreeFunction": 0.5, "Age": 33}
    ranges = {"Pregnancies": (0, 17), "Glucose": (0, 200), "BloodPressure": (0, 122),
              "SkinThickness": (0, 99), "Insulin": (0, 846), "BMI": (0.0, 67.0),
              "DiabetesPedigreeFunction": (0.0, 2.5), "Age": (21, 81)}
    for i, feat in enumerate(FEATURES):
        lo, hi = ranges[feat]
        with cols[i % 4]:
            if isinstance(defaults[feat], float):
                values[feat] = st.slider(feat, float(lo), float(hi), float(defaults[feat]))
            else:
                values[feat] = st.slider(feat, int(lo), int(hi), int(defaults[feat]))
    if st.button("Predict", type="primary"):
        out = predict_single(model, values)
        st.success(f"**{out['label']}** — probability of diabetes: {out['probability_diabetes']:.2%} "
                   f"(risk band: {out['risk_band']})")

    st.divider()
    st.subheader("Batch prediction (upload a CSV with the 8 feature columns)")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        try:
            scored = predict_batch(model, batch)
            st.dataframe(scored, use_container_width=True)
            st.download_button("Download predictions", scored.to_csv(index=False),
                               file_name="diabetes_predictions.csv", mime="text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
