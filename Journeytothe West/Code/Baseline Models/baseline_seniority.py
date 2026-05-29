from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = Path("train_set_seniority.csv")   # adjust if needed
OUTPUT_CSV = "seniority_baseline_predictions.csv"

# Keyword patterns 
SENIOR_PATS = [r"senior", r"\blead\b", r"principal", r"director", r"chief", r"head\b", r"vp\b", r"sr\."]
EXP_PATS    = [r"experienced", r"seasoned", r"5\+?\s*years", r"expert", r"consultant", r"specialist"]
INT_PATS    = [r"intermediate", r"mid[-\s]?level", r"associate", r"2\+?\s*years", r"3\+?\s*years"]
ENTRY_PATS  = [r"entry\s*level", r"junior", r"graduate", r"trainee", r"intern", r"assistant\b", r"no\s+experience"]

senior_re = re.compile("|".join(SENIOR_PATS), re.I)
exp_re    = re.compile("|".join(EXP_PATS),    re.I)
int_re    = re.compile("|".join(INT_PATS),    re.I)
entry_re  = re.compile("|".join(ENTRY_PATS),  re.I)

# Prediction 
def predict(text: str) -> str:
    if not isinstance(text, str):
        return "intermediate"
    t = text.lower()
    if senior_re.search(t):
        return "senior"
    if exp_re.search(t):
        return "experienced"
    if entry_re.search(t):
        return "entry level"
    if int_re.search(t):
        return "intermediate"
    return "intermediate"

# Load & run

df = pd.read_csv(DATA_PATH, encoding="latin1")
merged = (df["job_title"].fillna("") + " " + df["job_summary"].fillna("") + " " + df["job_ad_details"].fillna(""))

df["pred"] = merged.apply(predict)

df["correct"] = df["pred"] == df["y_true"].str.lower()

acc   = accuracy_score(df["y_true"].str.lower(), df["pred"])
macro_f1 = f1_score(df["y_true"].str.lower(), df["pred"], average="macro", zero_division=0)
print(f"Accuracy : {acc:.3f}\nMacro‑F1 : {macro_f1:.3f}")

cols = ["job_id", "y_true", "pred", "correct"]
df[cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"Saved predictions to {OUTPUT_CSV}")
