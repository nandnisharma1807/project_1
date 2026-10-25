"""Prediction backend used by the Streamlit frontend.

Loads models/best_model.pkl (a Pipeline: median imputer -> scaler
(where used) -> classifier) so the frontend never has to know which
model won, or repeat any preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def load_model(path: str | Path | None = None):
    if path:
        return joblib.load(Path(path))
    pkl = ROOT / "models" / "best_model.pkl"
    if pkl.exists():
        return joblib.load(pkl)
    # GitHub-friendly fallback: the trained model is also shipped as
    # base64 text (models/best_model.pkl.b64), because the GitHub file
    # API used for upload handles text, not raw binary.
    import base64
    import io

    b64 = ROOT / "models" / "best_model.pkl.b64"
    if b64.exists():
        return joblib.load(io.BytesIO(base64.b64decode(b64.read_text())))
    raise FileNotFoundError("No trained model found. Run `python backend/train.py` first.")


def predict_single(model, values: dict) -> dict:
    """values: dict with all FEATURES. Returns label, probability, risk band."""
    row = pd.DataFrame([{f: values[f] for f in FEATURES}])
    pred = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else None
    band = "High" if (proba is not None and proba >= 0.70) else "Moderate" if (proba is not None and proba >= 0.40) else "Low"
    return {"prediction": pred, "label": "Diabetes" if pred == 1 else "No diabetes",
            "probability_diabetes": proba, "risk_band": band}


def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [f for f in FEATURES if f not in out.columns]
    if missing:
        raise ValueError(f"Batch file is missing columns: {missing}")
    out["prediction"] = model.predict(out[FEATURES])
    if hasattr(model, "predict_proba"):
        out["probability_diabetes"] = model.predict_proba(out[FEATURES])[:, 1]
    out["label"] = out["prediction"].map({1: "Diabetes", 0: "No diabetes"})
    return out
