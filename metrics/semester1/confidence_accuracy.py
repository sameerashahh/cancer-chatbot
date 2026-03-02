import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------
with open("../basic_pipeline/model_results.json") as f:
    results = json.load(f)

with open("evaluation_results_with_confidence.json") as f:
    confidence_data = json.load(f)

df_results = pd.DataFrame(results)
df_confidence = pd.DataFrame(confidence_data)

# ----------------------------
# Merge datasets
# ----------------------------
if "question_id" in df_results.columns and "question_id" in df_confidence.columns:
    df = pd.merge(df_results, df_confidence, on="question_id", how="inner")
else:
    # fallback: align by index if no question_id
    df = pd.concat([df_results.reset_index(drop=True), df_confidence.reset_index(drop=True)], axis=1)

# Ensure relevant columns
df = df[["correct", "confidence"]].dropna()

# ----------------------------
# Bin by confidence into 5 levels (like EUREQA bins)
# ----------------------------
num_bins = 5
df["confidence_bin"] = pd.qcut(
    df["confidence"], 
    q=num_bins, 
    duplicates="drop"
)


# Compute mean accuracy per confidence bin
grouped = df.groupby("confidence_bin", observed=True).agg({
    "confidence": "mean",
    "correct": "mean"
}).reset_index()

# Compute correlation between confidence and accuracy
corr, pval = pearsonr(grouped["confidence"], grouped["correct"])

print("=== Confidence vs Accuracy Analysis ===")
print(grouped)
print(f"\nCorrelation between confidence and accuracy: {corr:.3f}")
print(f"P-value: {pval:.4f}")