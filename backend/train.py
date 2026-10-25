"""Train multiple models, in multiple iterations, and pick the best by F1.

Iterations (all on the same stratified 80/20 split, random_state=42,
plus 5-fold cross-validated F1 on the training set):
  1. Logistic Regression (scaled, median-imputed)
  2. Logistic Regression, class_weight='balanced'
  3. Random Forest
  4. Random Forest, tuned (small GridSearchCV)
  5. Gradient Boosting
  6. Support Vector Machine, RBF kernel (scaled)
  7. K-Nearest Neighbours (scaled)
  8. Decision Tree

Selection rule: highest test F1-score. Recall is reported alongside
because, for diabetes screening, missing a true case (false negative)
is the costly error. Ties are broken by higher recall, then accuracy.

Outputs (relative to the project root):
  models/best_model.pkl      - the winning Pipeline (imputer+scaler+model)
  models/best_model.json     - name + metrics of the winner
  results/model_comparison.csv
  results/results.json       - full run, incl. confusion matrices
  results/confusion_matrix_best.png
  data/diabetes_clean.csv    - cleaned data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.data_cleaning import ZERO_AS_MISSING, clean, load_raw  # noqa: E402

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
TARGET = "Outcome"


def make_pipeline(model, scale: bool = True) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate(name, pipe, X_train, y_train, X_test, y_test) -> dict:
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    cm = confusion_matrix(y_test, pred).tolist()
    row = {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4) if proba is not None else None,
        "cv_f1_mean": round(float(cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()), 4),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }
    return row


def main() -> None:
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)

    raw = load_raw(str(ROOT / "data" / "diabetes.csv"))
    cleaned, clean_report = clean(raw)
    cleaned.to_csv(ROOT / "data" / "diabetes_clean.csv", index=False)

    X = cleaned[FEATURES]
    y = cleaned[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = [
        ("Logistic Regression", make_pipeline(LogisticRegression(max_iter=1000))),
        ("Logistic Regression (balanced)", make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))),
        ("Random Forest", make_pipeline(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), scale=False)),
        ("Gradient Boosting", make_pipeline(GradientBoostingClassifier(random_state=42), scale=False)),
        ("SVM (RBF)", make_pipeline(SVC(probability=True, random_state=42))),
        ("KNN (k=11)", make_pipeline(KNeighborsClassifier(n_neighbors=11))),
        ("Decision Tree", make_pipeline(DecisionTreeClassifier(random_state=42), scale=False)),
    ]

    results, fitted = [], {}
    for name, pipe in candidates:
        row = evaluate(name, pipe, X_train, y_train, X_test, y_test)
        results.append(row)
        fitted[name] = pipe
        print(f"{name:35s} acc={row['accuracy']:.4f} prec={row['precision']:.4f} "
              f"recall={row['recall']:.4f} F1={row['f1']:.4f} ROC-AUC={row['roc_auc']} CV-F1={row['cv_f1_mean']:.4f}")

    # Tuned iteration: Random Forest grid search (the strongest family on
    # this dataset in iteration 1 in most runs; grid is kept small on purpose).
    grid = GridSearchCV(
        make_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1), scale=False),
        param_grid={
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 8, 12],
            "model__min_samples_leaf": [1, 2],
        },
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )
    tuned_row = evaluate("Random Forest (tuned)", grid, X_train, y_train, X_test, y_test)
    tuned_row["best_params"] = {k: str(v) for k, v in grid.best_params_.items()}
    results.append(tuned_row)
    fitted["Random Forest (tuned)"] = grid.best_estimator_
    print(f"{'Random Forest (tuned)':35s} acc={tuned_row['accuracy']:.4f} prec={tuned_row['precision']:.4f} "
          f"recall={tuned_row['recall']:.4f} F1={tuned_row['f1']:.4f} CV-F1={tuned_row['cv_f1_mean']:.4f}")

    # Best by F1, then recall, then accuracy.
    best = max(results, key=lambda r: (r["f1"], r["recall"], r["accuracy"]))
    best_pipe = fitted[best["model"]]
    joblib.dump(best_pipe, ROOT / "models" / "best_model.pkl")

    # Feature importance / coefficients for the Streamlit app.
    model_step = best_pipe.named_steps["model"] if hasattr(best_pipe, "named_steps") else best_pipe
    importance = None
    if hasattr(model_step, "feature_importances_"):
        importance = dict(zip(FEATURES, [round(float(v), 4) for v in model_step.feature_importances_]))
    elif hasattr(model_step, "coef_"):
        importance = dict(zip(FEATURES, [round(float(v), 4) for v in model_step.coef_[0]]))

    comparison = pd.DataFrame(
        [{k: r[k] for k in ("model", "accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1_mean")} for r in results]
    ).sort_values(["f1", "recall", "accuracy"], ascending=False)
    comparison.to_csv(ROOT / "results" / "model_comparison.csv", index=False)

    # Confusion-matrix plot for the winner.
    cm = np.array(best["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted: No diabetes", "Predicted: Diabetes"],
                yticklabels=["Actual: No diabetes", "Actual: Diabetes"])
    ax.set_title(f"Confusion Matrix — {best['model']}")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "confusion_matrix_best.png", dpi=150)
    plt.close(fig)

    payload = {
        "best_model": best["model"],
        "best_metrics": {k: best[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1_mean")},
        "confusion_matrix_best": best["confusion_matrix"],
        "feature_importance_or_coefficients": importance,
        "features": FEATURES,
        "target": TARGET,
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "cleaning_report": clean_report,
        "all_results": results,
    }
    (ROOT / "results" / "results.json").write_text(json.dumps(payload, indent=2))
    (ROOT / "models" / "best_model.json").write_text(json.dumps({
        "best_model": best["model"],
        "metrics": payload["best_metrics"],
        "features": FEATURES,
        "feature_importance_or_coefficients": importance,
    }, indent=2))
    print(f"\nBEST MODEL: {best['model']}  F1={best['f1']:.4f}  Recall={best['recall']:.4f}  "
          f"Accuracy={best['accuracy']:.4f}  Confusion matrix (TN,FP/FN,TP)={best['confusion_matrix']}")


if __name__ == "__main__":
    main()
