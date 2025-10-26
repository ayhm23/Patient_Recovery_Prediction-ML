# src/preprocess_data.py


import os
import pandas as pd
from datetime import datetime
from utils import feature_engineering

OUTDIR = "./data"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_save(df, filename):
    path = os.path.join(OUTDIR, filename)
    try:
        df.to_csv(path, index=False)
        print(f"Saved: {path}")
    except PermissionError:
        alt = os.path.join(OUTDIR, f"{filename.replace('.csv', f'_{TIMESTAMP}.csv')}")
        df.to_csv(alt, index=False)
        print(f"PermissionError, saved as: {alt}")

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    train_path = os.path.join(OUTDIR, "train.csv")
    test_path = os.path.join(OUTDIR, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(" Missing train.csv or test.csv in ./data/. Please place them there first.")
        return

    print(" Loading train/test data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("Applying feature engineering...")
    train_fe = feature_engineering(train)
    test_fe = feature_engineering(test)

    # Save processed data
    safe_save(train_fe, "train_preprocessed.csv")
    safe_save(test_fe, "test_preprocessed.csv")

    print("\n Preprocessing completed successfully.")
    print(f"Files are available in {os.path.abspath(OUTDIR)}")

if __name__ == "__main__":
    main()
