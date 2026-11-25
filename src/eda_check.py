from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv", low_memory=False)

print("=== 1. BASIC SHAPE ===")
print(f"Total rows (visits): {len(df)}")
print(f"Unique patients:     {df['patient_nbr'].nunique()}")

print("\n=== 2. TOP MISSING COLUMNS (NaN or '?') ===")
missing_pct = ((df.isna()) | (df == "?")).mean() * 100
print(missing_pct[missing_pct > 0].sort_values(ascending=False).round(2))

print("\n=== 3. DEAD / HOSPICE VISITS ===")
dead_hospice_ids = [11, 13, 14, 19, 20, 21]
dead_count = df["discharge_disposition_id"].isin(dead_hospice_ids).sum()
print(f"Rows to remove (expired/hospice): {dead_count}")

print("\n=== 4. CONSTANT COLUMNS (only 1 unique value) ===")
constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
print(constant_cols)

print("\n=== 5. BINARY TARGET BALANCE (<30 vs rest) ===")
binary_target = (df["readmitted"] == "<30").astype(int)
print(binary_target.value_counts(normalize=True).mul(100).round(2))