import os
import json
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def train_model():
    params = load_params()
    target_col = params["preprocessing"]["target_col"]
    xgb_params = params["model"]["xgboost"]

    # 1. Load processed splits
    train_df = pd.read_csv(params["data"]["train_path"])
    test_df = pd.read_csv(params["data"]["test_path"])

    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    # 2. Identify numeric vs categorical columns automatically
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    # 3. Build preprocessing transformers
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    # 4. Handle class imbalance (~9% positive class)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # 5. Combine preprocessor + model into ONE production pipeline
    model = XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # 6. Train & track with MLflow
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run():
        print("Training XGBoost pipeline...")
        pipeline.fit(X_train, y_train)

        # Predict probabilities for AUC metrics, and class labels for F1
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "pr_auc": round(average_precision_score(y_test, y_prob), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4)
        }

        print(f"Evaluation Metrics: {metrics}")

        # Log parameters, metrics, and model to MLflow
        mlflow.log_params(xgb_params)
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 2))
        mlflow.log_metrics(metrics)
        # mlflow.sklearn.log_model(pipeline, name="xgboost_pipeline")
        mlflow.sklearn.log_model(
            pipeline,
            name="xgboost_pipeline",
            serialization_format="cloudpickle"
        )

    # 7. Save model & scores locally for DVC and FastAPI
    os.makedirs(params["model"]["model_dir"], exist_ok=True)
    joblib.dump(pipeline, params["model"]["model_path"])

    with open(params["model"]["scores_path"], "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved to {params['model']['model_path']}")
    print(f"Scores saved to {params['model']['scores_path']}")

if __name__ == "__main__":
    train_model()