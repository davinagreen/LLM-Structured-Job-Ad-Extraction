from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = Path("train_set_salary.csv")  
OUTPUT_CSV = "salary_baseline_predictions.csv"

# Currency dictionaries
CURRENCY_MAP: dict[str, str] = {
    "$": "usd", "usd": "usd",
    "aud": "aud", "a$": "aud", "au$": "aud", "aud$": "aud",
    "nzd": "nzd", "nz$": "nzd",
    "cad": "cad", "c$": "cad",
    "sgd": "sgd", "sg$": "sgd",
    "hkd": "hkd", "hk$": "hkd",
    "myr": "myr", "rm":  "myr",
    "inr": "inr", "₹":   "inr",
    "php": "php", "₱":   "php",
    "idr": "idr", "rp":  "idr",
    "thb": "thb", "฿":   "thb",
    "gbp": "gbp", "£":   "gbp",
    "eur": "eur", "€":   "eur",
    "cny": "cny", "rmb": "cny", "¥": "cny",
    "krw": "krw", "₩":   "krw",
    "jpy": "jpy",
}

DEFAULT_BY_NATION = {
    "aus": "aud", "au": "aud",
    "nz": "nzd",
    "us": "usd",
    "ca": "cad",
    "sg": "sgd",
    "hk": "hkd",
    "my": "myr",
    "uk": "gbp", "gb": "gbp",
    "cn": "cny",
    "jp": "jpy",
    "kr": "krw",
    "ph": "php",
    "id": "idr",
    "th": "thb",
    "in": "inr",
}

# Frequency dictionaries
FREQ_CANONICAL = {
    "hour": "hourly", "hourly": "hourly", "per hour": "hourly", "ph": "hourly", "p/h": "hourly",
    "day": "daily", "daily": "daily", "per day": "daily",
    "week": "weekly", "weekly": "weekly", "per week": "weekly", "pw": "weekly", "p/w": "weekly",
    "month": "monthly", "monthly": "monthly", "per month": "monthly", "pm": "monthly", "p/m": "monthly",
    "year": "annual", "annum": "annual", "yearly": "annual", "annual": "annual",
    "pa": "annual", "p.a": "annual", "p.a.": "annual", "per annum": "annual",
}

FREQ_PRIORITY = ["hourly", "daily", "weekly", "monthly", "annual"]
FREQ_PATTERN = re.compile("|".join(map(re.escape, sorted(FREQ_CANONICAL.keys(), key=len, reverse=True))), re.I)

# Amount & range regex
NUMBER = r"\d{1,3}(?:,?\d{3})*(?:\.\d+)?(?:[kKmM])?"  # e.g., 1,500 | 85k | 1.2m
CUR = r"(?:[A-Za-z]{2,4}|[$€£¥₩₱฿₹])"  

RANGE_RE = re.compile(
    rf"(?:(?P<precur>{CUR})\s*)?"      # optional leading currency
    rf"(?P<min>{NUMBER})\s*"           # min amount
    rf"(?:-|–|—|to)\s*"                # range separator
    rf"(?:(?P<postcur>{CUR})\s*)?"     # optional second currency
    rf"(?P<max>{NUMBER})",
    flags=re.I,
)

SINGLE_RE = re.compile(
    rf"(?:(?P<cur>{CUR})\s*)?(?P<amt>{NUMBER})",
    flags=re.I,
)

# Helper functions
def _num_to_int(txt: str) -> int:
    txt = txt.replace(",", "").lower()
    mult = 1_000 if txt.endswith("k") else 1_000_000 if txt.endswith("m") else 1
    txt = txt.rstrip("km")
    try:
        return int(float(txt) * mult)
    except ValueError:
        return 0

def _norm_currency(raw: str | None, nation: str | None) -> str:
    if raw:
        code = CURRENCY_MAP.get(raw.lower().strip())
        if code:
            return code
    if nation:
        return DEFAULT_BY_NATION.get(nation.lower()[:2], "none")
    return "none"

def _detect_frequency(text: str) -> str:
    matches = [FREQ_CANONICAL[m.group(0).lower()] for m in FREQ_PATTERN.finditer(text)]
    if not matches:
        return "none"
    for f in FREQ_PRIORITY:
        if f in matches:
            return f
    return matches[0]

def extract_salary(text: str, nation: str | None = None) -> str:
    if not isinstance(text, str):
        return "0-0-none-none"
    txt = text.lower()

    m = RANGE_RE.search(txt)
    if m:
        cur_raw = m.group("precur") or m.group("postcur")
        cur = _norm_currency(cur_raw, nation)
        min_amt = _num_to_int(m.group("min"))
        max_amt = _num_to_int(m.group("max"))
    else:
        m2 = SINGLE_RE.search(txt)
        if not m2:
            return "0-0-none-none"
        cur = _norm_currency(m2.group("cur"), nation)
        min_amt = max_amt = _num_to_int(m2.group("amt"))

    freq = _detect_frequency(txt)
    return f"{min_amt}-{max_amt}-{cur}-{freq}"

#Load data & predict
df = pd.read_csv(DATA_PATH, encoding="latin1")
combined = (
    df["salary_additional_text"].fillna("") + " " +
    df["job_title"].fillna("") + " " +
    df["job_ad_details"].fillna("")
)

df["pred"] = [extract_salary(t, nation) for t, nation in zip(combined, df.get("nation_short_desc"))]

df["correct"] = df["pred"] == df["y_true"]

acc = accuracy_score(df["y_true"], df["pred"])
macro = f1_score(df["y_true"], df["pred"], average="macro", zero_division=0)
print(f"Accuracy : {acc:.3f}\nMacro-F1 : {macro:.3f}")

df[["job_id", "y_true", "pred", "correct"]].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"Saved predictions to {OUTPUT_CSV}")
