# project_1 — Diabetes Analytics & Prediction (Streamlit)

Cleaned diabetes data, multiple machine-learning models compared on the
same split, the best model chosen by F1-score, and a Streamlit app for
exploration and prediction.

> Analytics / screening-support tool only — not a medical diagnosis.

## Note on the dataset

No file was attached to the chat where this project was requested, so
this build uses the canonical Pima Indians Diabetes dataset
(768 rows, 8 features + `Outcome`), saved as `data/diabetes.csv`.
If you send your own diabetes CSV with the same 9 columns, drop it in
as `data/diabetes.csv` and re-run training — the whole pipeline,
including the model comparison, will rerun on your data.

## Cleaning (`backend/data_cleaning.py`)

- 768 rows, 0 duplicates, no rows dropped.
- Physiologically impossible zeros treated as missing, then median-filled:
  - Glucose: 5 zeros → median 117.0
  - BloodPressure: 35 zeros → median 72.0
  - SkinThickness: 227 zeros → median 29.0
  - Insulin: 374 zeros → median 125.0
  - BMI: 11 zeros → median 32.3
- Class balance after cleaning: 500 no-diabetes, 268 diabetes.
- Cleaned data: `data/diabetes_clean.csv`
- In training, imputation/scaling live inside an sklearn Pipeline fitted
  on the training split only, to avoid leakage.

## Model iterations — same stratified 80/20 split (seed 42, test n = 154)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression (balanced) — BEST** | **0.7338** | **0.6032** | **0.7037** | **0.6496** | **0.8126** | **0.6648** |
| Gradient Boosting | 0.7532 | 0.6818 | 0.5556 | 0.6122 | 0.8309 | 0.6424 |
| Random Forest (tuned) | 0.7468 | 0.6667 | 0.5556 | 0.6061 | 0.8157 | 0.6453 |
| KNN (k=11) | 0.7338 | 0.6327 | 0.5741 | 0.6019 | 0.7867 | 0.6257 |
| SVM (RBF) | 0.7403 | 0.6522 | 0.5556 | 0.6000 | 0.7964 | 0.6230 |
| Random Forest | 0.7338 | 0.6444 | 0.5370 | 0.5859 | 0.8158 | 0.6390 |
| Logistic Regression | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 | 0.6497 |
| Decision Tree | 0.6818 | 0.5532 | 0.4815 | 0.5149 | 0.6357 | 0.5715 |

Selection rule: highest test F1, then recall, then accuracy. Recall is
emphasised because missing a true diabetes case is the costly error.

### Best model — confusion matrix (test set)

```
                Predicted: No diabetes   Predicted: Diabetes
Actual: No diabetes          75 (TN)              25 (FP)
Actual: Diabetes             16 (FN)              38 (TP)
```

- Recall = 38 / (38 + 16) = 0.7037
- Precision = 38 / (38 + 25) = 0.6032
- F1 = 0.6496, Accuracy = 0.7338

Top coefficients (best model): Glucose 1.1834, BMI 0.7097,
Pregnancies 0.3730, DiabetesPedigreeFunction 0.2877, Age 0.1864.
Full comparison: `results/model_comparison.csv`, `results/results.json`.
Confusion-matrix plot: `results/confusion_matrix_best.png`.

## Structure

```
project_1/
├── backend/
│   ├── data_cleaning.py   # backend: cleaning rules + report
│   ├── train.py           # backend: 8 model iterations, pick best, save artefacts
│   └── predict.py         # backend: load best model, single + batch prediction
├── frontend/
│   └── app.py             # frontend: Streamlit app (Overview / Model Comparison / Predict)
├── data/
│   ├── diabetes.csv       # raw (canonical Pima dataset, 768 rows)
│   └── diabetes_clean.csv # cleaned
├── models/
│   ├── best_model.pkl     # winning Pipeline (local runs)
│   ├── best_model.pkl.b64 # same model as base64 text, for GitHub upload
│   └── best_model.json    # winner name + metrics
├── results/
│   ├── model_comparison.csv
│   ├── results.json
│   └── confusion_matrix_best.png
└── requirements.txt
```

## Run it locally

```bash
pip install -r requirements.txt
python backend/train.py          # retrains, re-picks the best model
streamlit run frontend/app.py    # opens the app (verified locally)
```

The app has three tabs: Overview & Data, Model Comparison (with the
confusion matrix), and Predict — a single-patient slider form plus
batch CSV prediction with a downloadable results file.

## Deploy to Streamlit Cloud

Push this repo, then in Streamlit Cloud choose this repository, branch
`main`, main file `frontend/app.py`. The app loads the model from
`models/best_model.pkl.b64` if no `.pkl` is present, so no retraining
step is needed on deploy.
