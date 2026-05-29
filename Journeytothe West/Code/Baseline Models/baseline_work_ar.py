from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = Path("train_set_work_arrangements.csv")  # adjust if needed
OUTPUT_CSV = "work_arr_baseline_predictions.csv"

REMOTE_KWS = [r"remote", r"work\s*from\s*home", r"wfh", r"fully\s*remote"]
HYBRID_KWS = [r"hybrid", r"flexible working", r"split\s*days", r"partly\s*remote"]

remote_re = re.compile("|".join(REMOTE_KWS), re.I)
hybrid_re = re.compile("|".join(HYBRID_KWS), re.I)

# Prediction rule
def predict(text: str) -> str:
    if not isinstance(text, str):
        return "onsite"
    if remote_re.search(text):
        return "remote"
    if hybrid_re.search(text):
        return "hybrid"
    return "onsite"

# Load data & predict
df = pd.read_csv(DATA_PATH, encoding="latin1")

df["pred"] = df["job_ad"].fillna("").str.lower().apply(predict)

df["correct"] = df["pred"] == df["y_true"].str.lower()

# Metrics
acc = accuracy_score(df["y_true"].str.lower(), df["pred"])
macro_f1 = f1_score(df["y_true"].str.lower(), df["pred"], average="macro", zero_division=0)
print(f"Accuracy : {acc:.3f}\nMacro‑F1 : {macro_f1:.3f}")

# Save results
cols = ["id", "y_true", "pred", "correct"]
df[cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"Saved predictions to {OUTPUT_CSV}")
