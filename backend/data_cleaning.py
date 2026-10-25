"""Data cleaning for the diabetes dataset.

Cleaning rules (documented, reproducible):
- Column names are standardised to the Pima Indians Diabetes layout.
- Physiologically impossible zeros are treated as missing, NOT as real
  measurements, in: Glucose, BloodPressure, SkinThickness, Insulin, BMI.
  Pregnancies, DiabetesPedigreeFunction, Age and Outcome can legitimately
  be zero, so they are left alone.
- Missing values are filled with the median (robust to outliers).
  In training, the median is fitted on the training split only, inside
  an sklearn Pipeline, to avoid leakage. This module's `clean()` is used
  for the saved exploratory / cleaned CSV, where the median is reported
  in the cleaning report.
- No rows are dropped: the dataset is small (768 rows), and dropping
  every row with a zero would throw away a large, biased share of it.
- Outcome is kept as an integer 0/1 target.
"""

from __future__ import annotations

import pandas as pd

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_raw(path: str) -> pd.DataFrame:
    """Load a header-less Pima-style CSV, or a CSV that already has headers."""
    df = pd.read_csv(path)
    if list(df.columns) == COLUMNS:
        return df
    if df.shape[1] == 9:
        df.columns = COLUMNS
        return df
    raise ValueError(f"Expected 9 columns, got {df.shape[1]}: {list(df.columns)}")


def cleaning_report(df: pd.DataFrame) -> dict:
    """Count impossible zeros before cleaning, per column."""
    return {c: int((df[c] == 0).sum()) for c in ZERO_AS_MISSING}


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (cleaned_df, report). Does not mutate the input."""
    report = {
        "rows_before": int(len(df)),
        "columns": list(df.columns),
        "impossible_zeros_before": cleaning_report(df),
        "duplicates_before": int(df.duplicated().sum()),
    }
    out = df.copy()
    out = out.drop_duplicates()
    for col in ZERO_AS_MISSING:
        out[col] = out[col].replace(0, pd.NA)
    medians = {c: float(out[c].median()) for c in ZERO_AS_MISSING}
    out[ZERO_AS_MISSING] = out[ZERO_AS_MISSING].fillna(medians)
    report["medians_used"] = medians
    report["rows_after"] = int(len(out))
    report["missing_after"] = int(out.isna().sum().sum())
    report["class_balance"] = out["Outcome"].value_counts().to_dict()
    # JSON-friendly keys
    report["class_balance"] = {str(k): int(v) for k, v in report["class_balance"].items()}
    return out, report


if __name__ == "__main__":
    import json
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data/diabetes.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/diabetes_clean.csv"
    raw = load_raw(src)
    cleaned, rep = clean(raw)
    cleaned.to_csv(dst, index=False)
    print(json.dumps(rep, indent=2))
    print(f"Saved cleaned data: {dst} ({len(cleaned)} rows)")
