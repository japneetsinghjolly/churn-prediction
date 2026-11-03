"""Generate suggested retention offers; this script does not send emails."""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SCORES_FILE = BASE_DIR / "scores.csv"
OUTPUT_FILE = BASE_DIR / "emails.csv"


def offer(probability: float) -> str:
    if probability > .80:
        return "Review eligibility for a 20% discount for 12 months"
    if probability > .70:
        return "Review eligibility for a 15% discount for 6 months"
    return "Review eligibility for a 10% discount for 3 months"


def main():
    if not SCORES_FILE.is_file():
        raise FileNotFoundError("scores.csv not found. Run `python score.py` first.")
    df = pd.read_csv(SCORES_FILE)
    if not {"risk", "churn_probability"}.issubset(df.columns):
        raise ValueError("scores.csv must contain risk and churn_probability columns.")
    high = df[df["risk"] == "HIGH"].copy()
    high["suggested_offer"] = high["churn_probability"].apply(offer)
    high.to_csv(OUTPUT_FILE, index=False)
    print(f"Prepared {len(high)} suggested offers in {OUTPUT_FILE.name}. No emails were sent.")


if __name__ == "__main__":
    main()
