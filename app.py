"""Customer churn risk scoring web app, configured for local use and Render."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURES_PATH = BASE_DIR / "features.txt"
DATA_PATH = BASE_DIR / "data" / "Telco_Churn.csv"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

model = None
features: list[str] = []
model_load_error: str | None = None

try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(f"Feature list not found: {FEATURES_PATH.name}")
    model = joblib.load(MODEL_PATH)
    features = [line.strip() for line in FEATURES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not features:
        raise ValueError("The feature list is empty.")
except Exception as exc:  # keep health route available so deployment errors are diagnosable
    model_load_error = f"{type(exc).__name__}: {exc}"
    app.logger.exception("Unable to load churn model")


CATEGORIES = {
    "InternetService": {"DSL", "Fiber optic", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "OnlineSecurity": {"No", "Yes", "No internet service"},
}
NUMERIC_LIMITS = {
    "tenure": (0, 72),
    "MonthlyCharges": (0, 200),
    "TotalCharges": (0, 100000),
}


def encode_customer(data: dict[str, Any]) -> pd.DataFrame:
    """Encode a single customer using the exact feature names used in training.

    Do not call get_dummies on a single row: it drops baseline/category columns
    depending on which values happen to appear in that row.
    """
    row: dict[str, float] = {}
    for name, (minimum, maximum) in NUMERIC_LIMITS.items():
        value = data.get(name)
        if value is None or value == "":
            raise ValueError(f"{name} is required.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number.") from None
        if not pd.notna(number) or number < minimum or number > maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        if name == "tenure" and not number.is_integer():
            raise ValueError("tenure must be a whole number of months.")
        row[name] = number

    for name, allowed in CATEGORIES.items():
        value = data.get(name)
        if value not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"{name} must be one of: {choices}.")
        for category in sorted(allowed):
            # These names mirror pandas.get_dummies(..., drop_first=True) on
            # the full training dataset, with the baseline categories omitted.
            if name == "InternetService" and category in {"Fiber optic", "No"}:
                row[f"InternetService_{category}"] = float(value == category)
            elif name == "Contract" and category in {"One year", "Two year"}:
                row[f"Contract_{category}"] = float(value == category)
            elif name == "OnlineSecurity" and category in {"No internet service", "Yes"}:
                row[f"OnlineSecurity_{category}"] = float(value == category)

    missing = [feature for feature in features if feature not in row]
    if missing:
        raise RuntimeError("Model feature mapping is incomplete: " + ", ".join(missing))
    return pd.DataFrame([[row[feature] for feature in features]], columns=features)


def risk_band(probability: float) -> tuple[str, str, str]:
    if probability >= 0.60:
        return "High", "high", "Prioritize a retention conversation and review service concerns."
    if probability >= 0.40:
        return "Medium", "medium", "Monitor engagement and consider a relevant check-in."
    return "Low", "low", "Continue the current customer experience and monitor over time."


def require_model():
    if model is None:
        return jsonify(error="The prediction model could not be loaded. Check the deployment logs and model.pkl."), 503
    return None


@app.get("/")
def home():
    return render_template("index.html", model_ready=model is not None, model_error=model_load_error)


@app.get("/health")
@app.get("/api/health")
def health():
    ready = model is not None and bool(features)
    status_code = 200 if ready else 503
    return jsonify(
        status="ok" if ready else "degraded",
        service="Customer Churn Intelligence",
        model_loaded=model is not None,
        feature_count=len(features),
        dataset_available=DATA_PATH.is_file(),
        error=model_load_error,
    ), status_code


@app.post("/predict")
@app.post("/api/predict")
def predict():
    unavailable = require_model()
    if unavailable:
        return unavailable
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send customer details as a JSON object."), 400
    try:
        encoded = encode_customer(data)
        probability = float(model.predict_proba(encoded)[0][1])
        label, css_class, recommendation = risk_band(probability)
        return jsonify(
            probability=probability,
            probability_percent=round(probability * 100, 1),
            risk=label,
            risk_key=css_class,
            recommendation=recommendation,
            model="XGBoost",
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction failed unexpectedly. Check application logs."), 500


@app.post("/api/batch-score")
def batch_score():
    """Score a CSV with the six fields used by the model; return downloadable CSV."""
    unavailable = require_model()
    if unavailable:
        return unavailable
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify(error="Choose a CSV file to score."), 400
    if not upload.filename.lower().endswith(".csv"):
        return jsonify(error="Upload a .csv file."), 400
    try:
        frame = pd.read_csv(upload, nrows=10001)
        if frame.empty:
            return jsonify(error="The uploaded CSV has no customer rows."), 400
        if len(frame) > 10000:
            return jsonify(error="Upload at most 10,000 customer rows per batch."), 400
        required = list(NUMERIC_LIMITS) + list(CATEGORIES)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            return jsonify(error="CSV is missing required columns: " + ", ".join(missing)), 400
        # Match the training-time treatment of blank TotalCharges values.
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
        median_total = frame["TotalCharges"].median()
        frame["TotalCharges"] = frame["TotalCharges"].fillna(0 if pd.isna(median_total) else median_total)
        rows = []
        for index, record in frame.iterrows():
            try:
                rows.append(encode_customer(record.to_dict()).iloc[0].to_dict())
            except ValueError as exc:
                return jsonify(error=f"Row {index + 2}: {exc}"), 400
        encoded = pd.DataFrame(rows, columns=features)
        frame["churn_probability"] = model.predict_proba(encoded)[:, 1]
        frame["risk"] = frame["churn_probability"].apply(lambda p: risk_band(float(p))[0])
        from io import BytesIO
        from flask import send_file
        output = BytesIO(frame.to_csv(index=False).encode("utf-8"))
        output.seek(0)
        return send_file(output, mimetype="text/csv", as_attachment=True, download_name="churn-risk-scores.csv")
    except pd.errors.EmptyDataError:
        return jsonify(error="The uploaded CSV is empty."), 400
    except Exception:
        app.logger.exception("Batch scoring failed")
        return jsonify(error="Batch scoring failed. Verify the CSV and review application logs."), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
