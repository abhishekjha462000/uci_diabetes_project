# 🏥 End-to-End 30-Day Diabetes Readmission Predictor

An end-to-end MLOps web application that predicts the risk of 30-day hospital readmission for diabetic patients using the UCI Diabetes 130-US Hospitals dataset.

🔗 **Live Demo:** [https://diabetes-readmission-predictor-epi2.onrender.com/](https://diabetes-readmission-predictor-epi2.onrender.com/)  
📄 **Interactive API Docs (Swagger):** [https://diabetes-readmission-predictor-epi2.onrender.com/docs](https://diabetes-readmission-predictor-epi2.onrender.com/docs)

---

## 🏗️ Architecture & MLOps Workflow

```text
Raw Data (UCI Diabetes)
   │
   ▼
[DVC Pipeline: preprocess] ──► Deduplication, Clinical NaN Handling, Stratified Split
   │
   ▼
[DVC Pipeline: train]      ──► Scikit-learn Pipeline + XGBoost ──► MLflow Tracking
   │
   ▼
models/model.joblib
   │
   ▼
[FastAPI + Tailwind UI]    ──► Dockerized Container ──► GitHub Actions CI ──► Render Cloud
```

### Tech Stack
* **Modeling & Pipelines:** Python, Pandas, Scikit-learn (`ColumnTransformer`, `Pipeline`), XGBoost
* **MLOps & Reproducibility:** DVC (`dvc.yaml`, `dvc.lock`), MLflow (experiment tracking), YAML-driven configs (`params.yaml`)
* **Serving & UI:** FastAPI, Pydantic (schema validation), Uvicorn, HTML/Tailwind CSS
* **Deployment & CI/CD:** Docker, GitHub Actions (`pytest` + automated Docker build check), Render

---

## 📊 Key Data Engineering & Model Metrics

* **Leakage Prevention:** Deduplicated `101,766` hospital encounters to `71,518` unique patients (first encounter) and removed `2,423` expired/hospice visits (`discharge_disposition_id`).
* **Clinical Signal Preservation:** Retained `A1Cresult` and `max_glu_serum` missingness as `"Not_Measured"` rather than dropping them.
* **Imbalance Handling:** Stratified train/test split (`8.98%` positive `<30` day readmission rate) with dynamic `scale_pos_weight` in XGBoost.

| Metric | Baseline XGBoost Score |
| :--- | :---: |
| **ROC-AUC** | `0.6548` |
| **PR-AUC** | `0.1815` |
| **F1-Score** | `0.2282` |

---

## 🚀 Run Locally

### Option 1: Using Docker (Recommended)
```bash
git clone [https://github.com/abhishekjha462000/uci_diabetes_project.git](https://github.com/abhishekjha462000/uci_diabetes_project.git)
cd uci_diabetes_project
docker build -t diabetes-readmission-app .
docker run -p 8000:8000 diabetes-readmission-app
```
Open `http://localhost:8000` in your browser.

### Option 2: Using Python Virtual Environment
```bash
git clone [https://github.com/abhishekjha462000/uci_diabetes_project.git](https://github.com/abhishekjha462000/uci_diabetes_project.git)
cd uci_diabetes_project
python -m venv venv
venv\Scripts\activate        # On Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# Reproduce the full DVC pipeline (optional)
dvc repro

# Launch the FastAPI app + UI
uvicorn app.api:app --reload
```

---

## 📂 Project Structure
```text
├── .github/workflows/ci.yml   # Automated pytest & Docker build CI
├── app/
│   ├── api.py                 # FastAPI backend & Pydantic schemas
│   └── static/index.html      # Dark-mode Tailwind CSS clinical UI
├── src/
│   ├── data_ingestion.py      # Raw dataset ingestion
│   ├── eda_check.py           # Exploratory data checks
│   ├── data_preprocessing.py  # Cleaning & stratified splitting
│   └── model_trainer.py       # Pipeline training + MLflow logging
├── tests/test_config.py       # Unit tests for CI
├── dvc.yaml & dvc.lock        # Reproducible pipeline DAG
├── params.yaml                # Centralized hyperparameters & paths
├── scores.json                # Tracked evaluation metrics
└── Dockerfile                 # Production container specification
```