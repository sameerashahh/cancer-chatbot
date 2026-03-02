import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ----------------------------------------
# Load JSON
# ----------------------------------------
with open("self_consistency_results.json", "r") as f:
    data = json.load(f)

# ----------------------------------------
# Build DataFrame
# ----------------------------------------
rows = []
for e in data:
    sc = e["self_consistency"]
    acc = 1 if e["majority_label"] == e["true_answer"] else 0
    rows.append({"self_consistency": sc, "accuracy": acc})

df = pd.DataFrame(rows)

# ----------------------------------------
# Normal Pearson correlation
# ----------------------------------------
pearson_corr, pearson_p = pearsonr(df["self_consistency"], df["accuracy"])

print("=== Overall Pearson Correlation ===")
print("Correlation:", pearson_corr)
print("p-value:", pearson_p)