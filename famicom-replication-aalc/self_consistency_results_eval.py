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

# ----------------------------------------
# Bin-wise correlation
# ----------------------------------------
bins = np.linspace(0, 1, 11)     # 0.0–1.0 in steps of 0.1
df["sc_bin"] = pd.cut(df["self_consistency"], bins)

# Accuracy per bin
bin_table = df.groupby("sc_bin")["accuracy"].mean()

# Bin centers
bin_centers = np.array([(interval.left + interval.right)/2 for interval in bin_table.index])
bin_accuracies = bin_table.values

# Correlation between bin centers and bin accuracies
bin_corr, bin_p = pearsonr(bin_centers, bin_accuracies)

print("\n=== Bin-wise Pearson Correlation ===")
print("Correlation:", bin_corr)
print("p-value:", bin_p)

# Optional: show table
print("\nBin Accuracies:")
print(pd.DataFrame({
    "bin_center": bin_centers,
    "bin_accuracy": bin_accuracies
}))
