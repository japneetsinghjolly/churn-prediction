"""Score all bundled customer records and save churn risk predictions."""
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "Telco_Churn.csv"
MODEL_FILE = BASE_DIR / "model.pkl"
FEATURES_FILE = BASE_DIR / "features.txt"
OUTPUT_FILE = BASE_DIR / "scores.csv"
COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "InternetService", "Contract", "OnlineSecurity"]


def main():
    for path in (DATA_FILE, MODEL_FILE, FEATURES_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required project file not found: {path}")
    df = pd.read_csv(DATA_FILE)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    X = pd.get_dummies(df[COLUMNS], columns=COLUMNS[3:], drop_first=True, dtype=int)
    features = [line.strip() for line in FEATURES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    for feature in features:
        if feature not in X.columns:
            X[feature] = 0
    X = X[features]
    model = joblib.load(MODEL_FILE)
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["risk"] = df["churn_probability"].map(lambda p: "HIGH" if p >= .60 else ("MEDIUM" if p >= .40 else "LOW"))
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE.name}")
    print(df["risk"].value_counts().to_string())


if __name__ == "__main__":
    main()
