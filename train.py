"""Train and evaluate the churn classifier using the bundled Telco dataset."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "Telco_Churn.csv"
MODEL_FILE = BASE_DIR / "model.pkl"
FEATURES_FILE = BASE_DIR / "features.txt"
COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "InternetService", "Contract", "OnlineSecurity"]


def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    required = COLUMNS + ["Churn"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

    df["Churn"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    encoded = pd.get_dummies(df[COLUMNS + ["Churn"]], columns=COLUMNS[3:], drop_first=True, dtype=int)
    X = encoded.drop(columns=["Churn"])
    y = encoded["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        random_state=42, eval_metric="logloss", n_jobs=1
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    print(f"Rows: {len(df):,} | Features: {X.shape[1]}")
    print(f"Accuracy:  {accuracy_score(y_test, predicted):.4f}")
    print(f"Precision: {precision_score(y_test, predicted, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predicted, zero_division=0):.4f}")
    print(f"F1 score:  {f1_score(y_test, predicted, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")
    print(classification_report(y_test, predicted, zero_division=0))

    joblib.dump(model, MODEL_FILE)
    FEATURES_FILE.write_text("\n".join(X.columns) + "\n", encoding="utf-8")
    print(f"Saved model: {MODEL_FILE.name}")
    print(f"Saved features: {FEATURES_FILE.name}")


if __name__ == "__main__":
    main()
