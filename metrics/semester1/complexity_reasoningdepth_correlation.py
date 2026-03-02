import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Load data
with open("../../basic_pipeline/model_results.json") as f:
    results = json.load(f)

with open("complexity_results.json") as f:
    complexity = json.load(f)

df_results = pd.DataFrame(results)
df_complexity = pd.DataFrame(complexity)

# Merge datasets
if "question_id" in df_results.columns and "question_id" in df_complexity.columns:
    df = pd.merge(df_results, df_complexity, on="question_id", how="inner")
else:
    df = pd.concat([df_results, df_complexity], axis=1)

# Ensure relevant columns
df = df[["correct", "length_complexity"]].dropna()

# ----------------------------
# EUREQA-style binning
# ----------------------------
# Bin by reasoning complexity into 5 discrete "depth" tiers (like depth 1–5)
num_bins = 5
df["reasoning_depth"] = pd.qcut(df["length_complexity"], num_bins, labels=[1, 2, 3, 4, 5])

# Compute accuracy per bin
grouped = df.groupby("reasoning_depth", observed=True).agg({
    "length_complexity": "mean",
    "correct": "mean"
}).reset_index()

# Compute correlation between bin averages
corr, pval = pearsonr(grouped["length_complexity"], grouped["correct"])

print("=== EUREQA-Style Reasoning Depth Analysis ===")
print(grouped)
print(f"\nCorrelation between reasoning depth (complexity) and accuracy: {corr:.3f}")
print(f"P-value: {pval:.4f}")

# ----------------------------
# Optional: visualize
# ----------------------------
plt.figure(figsize=(7, 4))
plt.plot(grouped["reasoning_depth"], grouped["correct"], marker="o", linewidth=2)
plt.xlabel("Reasoning Depth (Complexity Bin 1–5)")
plt.ylabel("Mean Accuracy")
plt.title("Accuracy vs Length of Tokens")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
