# RetentionIQ — Customer Churn Intelligence

A polished, responsive Flask web app for estimating customer churn risk with an included XGBoost model and the Telco Customer Churn dataset. Includes a single-customer assessment UI, JSON prediction API, health endpoint, optional batch CSV scoring endpoint, and Render deployment configuration.

> **Important:** This is a demonstration model. Its output is an estimate, not a guarantee of customer behaviour. Validate performance on your own data before using scores for business decisions. The included model uses six selected predictors: tenure, monthly charges, total charges, internet service, contract type, and online security.

## Features

- Premium responsive dashboard with customer-risk visualisation and suggested follow-up actions.
- Input validation on both browser and server.
- Correct categorical encoding for single-customer predictions (does not infer dummy columns from a single row).
- `GET /health` and `GET /api/health` health checks for deployment monitoring.
- `POST /predict` and `POST /api/predict` JSON prediction endpoints.
- `POST /api/batch-score` accepts a CSV containing the six required feature columns and returns a scored CSV (up to 10,000 rows per request).
- Gunicorn production server configuration and Render Blueprint (`render.yaml`).
- Unit tests for input validation, encoding, health checks, and prediction responses.

## Project structure

```text
churn-prediction/
├── app.py
├── model.pkl
├── features.txt
├── requirements.txt
├── render.yaml
├── Procfile
├── .python-version
├── train.py
├── score.py
├── suggest_offers.py
├── data/
│   └── Telco_Churn.csv
├── templates/
│   └── index.html
├── static/
│   ├── css/styles.css
│   └── js/app.js
└── tests/test_app.py
```

## Run locally

Python 3.12 is recommended for the included model artifact.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Test the project

```bash
python -m unittest discover -s tests -v
```

Check service health at `http://127.0.0.1:5000/health`.

## Prediction API example

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure":12,"MonthlyCharges":70,"TotalCharges":840,"InternetService":"Fiber optic","Contract":"Month-to-month","OnlineSecurity":"No"}'
```

Accepted values:
- `InternetService`: `DSL`, `Fiber optic`, `No`
- `Contract`: `Month-to-month`, `One year`, `Two year`
- `OnlineSecurity`: `No`, `Yes`, `No internet service`
- `tenure`: 0–72 months
- `MonthlyCharges`: 0–200
- `TotalCharges`: 0–100000

## Deploy to Render

### Option A — Blueprint

1. Push this project folder to a GitHub repository.
2. In Render, choose **New + → Blueprint** and connect the repository.
3. Render reads `render.yaml`, installs `requirements.txt`, starts Gunicorn, and checks `/health`.
4. After deployment, open the generated `.onrender.com` URL and confirm the dashboard loads.
5. Open `/health`; a healthy response includes `"status":"ok"`, `"model_loaded":true`, and `"feature_count":9`.

### Option B — Create a Web Service manually

- **Root directory:** `churn-prediction` if this is inside a larger monorepo; otherwise leave blank.
- **Runtime:** Python 3.
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app --workers 1 --threads 4 --timeout 120`
- **Health check path:** `/health`
- **Python version:** 3.12.8 (configured in `.python-version` and `render.yaml`).

The model and feature list must be committed alongside `app.py`. Do not put them in a temporary directory. Free web services may sleep when inactive and take time to wake up.

## Retrain the model

The included `train.py` reads `data/Telco_Churn.csv`, trains the XGBoost classifier, prints evaluation metrics, and regenerates `model.pkl` and `features.txt`.

```bash
python train.py
```

Retraining overwrites the included model artifacts. Save a backup first if you want to preserve the original model.

## Batch scoring and suggested offers

```bash
python score.py
python suggest_offers.py
```

These scripts create `scores.csv` and `emails.csv`. The offer script only generates suggested offers; it does not send emails.

## Data and privacy

The single-customer prediction endpoint does not save submitted form inputs to a database. Standard hosting logs may still record request metadata. Do not submit real customer personal information to a public demo. The batch endpoint processes the uploaded CSV in memory and returns a scored file; keep uploads free of unnecessary personal identifiers.
