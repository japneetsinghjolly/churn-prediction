# Churn Prediction

Simple project to predict which customers will leave.

## Setup

1. Install: `pip install -r requirements.txt`
2. Download data from Kaggle (put in folder)
3. Run: `python train.py`
4. Run: `python app.py`
5. Visit: http://localhost:5000

## Files

- `train.py` - trains model
- `app.py` - runs website
- `score.py` - predicts for all customers

## How it works

1. Read customer data (7000+ customers)
2. Train XGBoost model to predict who leaves
3. Get predictions from website
4. Score all customers automatically
5. Send offers to high-risk customers

Done!
