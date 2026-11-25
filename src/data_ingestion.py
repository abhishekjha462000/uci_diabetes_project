from pathlib import Path
from ucimlrepo import fetch_ucirepo


def ingest_data():
    # 1. Define where the raw data should be saved
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)  # Creates the folder if it doesn't exist
    output_path = raw_dir / "diabetic_data.csv"

    # 2. Fetch the dataset from UCI
    print("Fetching UCI Diabetes dataset (id=296)...")
    ds = fetch_ucirepo(id=296)
    df = ds.data.original  # features + target combined

    # 3. Save locally
    df.to_csv(output_path, index=False)
    print(f"Saved successfully to {output_path}")
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target distribution:\n{df['readmitted'].value_counts()}")


if __name__ == "__main__":
    ingest_data()