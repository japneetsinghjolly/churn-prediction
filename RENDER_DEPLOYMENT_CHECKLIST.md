# Render deployment checklist

1. Extract `RetentionIQ_Churn_Render_Ready.zip`.
2. Create a GitHub repository and upload the **contents** of `retentioniq-churn-predictor` to its root.
3. In Render, choose **New + → Blueprint**, connect the repository, and deploy the service defined by `render.yaml`.
4. If creating the service manually, use:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 1 --threads 4 --timeout 120`
   - Health check path: `/health`
5. When deployment finishes, open the Render URL. Then open `/health`; expect `status: ok`, `model_loaded: true`, and `feature_count: 9`.
6. Run a test prediction in the form. If the health check fails, open Render → service → Logs and verify that `app.py`, `model.pkl`, and `features.txt` are at the repository root.

The project has been smoke-tested locally for Python syntax, JavaScript syntax, model loading, feature mapping, predictions, and batch scoring against the bundled dataset. Run `python -m unittest discover -s tests -v` after installing dependencies to execute the Flask route tests.
