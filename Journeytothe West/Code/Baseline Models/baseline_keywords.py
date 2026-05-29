import pandas as pd, re, itertools
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

# Load and preprocess data
path = "train_set_job_keyword.csv"
df = pd.read_csv(path, encoding="latin1")
df["text"] = (df["job_title"].fillna("") + " " +
              df["job_ad_details"].fillna("")).str.lower()

# Tokenization setup
token_re = re.compile(r"[a-zA-Z']+")
stop = {"the","and","to","of","a","in","for","with","on","be","as",
        "we","you","our","your","is","are","or","an","will","this",
        "that","by","from","have","has","it","us"}
def tokenize(t): return [w for w in token_re.findall(t) if w not in stop]

# Extract top keywords per label
TOP_K = 10
label_keywords = {}
for label, g in df.groupby("job_keyword_true"):
    toks = list(itertools.chain.from_iterable(g["text"].map(tokenize)))
    label_keywords[label] = {w for w, _ in Counter(toks).most_common(TOP_K)}

# Prediction using keyword matching
def predict(text):
    toks = set(tokenize(text))
    hits = {lbl: len(toks & kws) for lbl, kws in label_keywords.items()}
    best_lbl, best_cnt = max(hits.items(), key=lambda x: x[1])
    return best_lbl if best_cnt else "unknown"

df["pred"] = df["text"].apply(predict)

# Evaluation metrics
df["correct"] = (df["pred"] == df["job_keyword_true"])
acc = accuracy_score(df["job_keyword_true"], df["pred"])
macro_f1 = f1_score(df["job_keyword_true"], df["pred"],
                    average="macro", zero_division=0)

print(f"Accuracy : {acc:.3f}")
print(f"Macro-F1 : {macro_f1:.3f}")

# Save results
df[["job_title", "job_keyword_true", "pred", "correct"]].to_csv(
    "keyword_baseline_predictions.csv",
    index=False, encoding="utf-8"
)
print("Saved predictions to keyword_baseline_predictions.csv")