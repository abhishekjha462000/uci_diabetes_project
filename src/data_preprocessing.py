import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_data():
    params = load_params()
    prep = params["preprocessing"]

    # 1. Load raw data
    df = pd.read_csv(params["data"]["raw_path"])
    print(f"Raw shape: {df.shape}")

    # 2. Replace '?' with NaN
    df = df.replace("?", np.nan)

    # 3. Preserve clinical signal: NaN in lab tests means 'Not_Measured'
    for col in prep["not_measured_cols"]:
        if col in df.columns:
            df[col] = df[col].fillna("Not_Measured")

    # 4. Remove expired / hospice visits
    df = df[~df["discharge_disposition_id"].isin(prep["dead_hospice_ids"])]

    # 5. Deduplicate by patient_nbr to prevent data leakage, then drop ID
    df = df.drop_duplicates(subset=["patient_nbr"], keep="first")
    df = df.drop(columns=["patient_nbr"])

    # 6. Drop useless / constant columns
    drop_cols = [c for c in prep["drop_cols"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # 7. Create binary target (1 = readmitted <30 days, 0 = otherwise)
    target_col = prep["target_col"]
    df[target_col] = (df[target_col] == prep["positive_class"]).astype(int)

    # 8. Stratified Train/Test Split
    train_df, test_df = train_test_split(
        df,
        test_size=prep["test_size"],
        random_state=prep["random_state"],
        stratify=df[target_col]
    )

    # 9. Save processed splits
    os.makedirs(params["data"]["processed_dir"], exist_ok=True)
    train_df.to_csv(params["data"]["train_path"], index=False)
    test_df.to_csv(params["data"]["test_path"], index=False)

    print(f"Train saved: {train_df.shape} | Positive rate: {train_df[target_col].mean():.4f}")
    print(f"Test saved:  {test_df.shape}  | Positive rate: {test_df[target_col].mean():.4f}")

if __name__ == "__main__":
    preprocess_data()