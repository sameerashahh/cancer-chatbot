import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ----------------------------
# Load both files
# ----------------------------
with open("phi_evaluation_results.json") as f:
    phi = json.load(f)

with open("../basic_pipeline/model_results.json") as f:
    acc = json.load(f)

df_phi = pd.DataFrame(phi)
df_acc = pd.DataFrame(acc)

# ----------------------------
# Index-based merge (no question_id)
# ----------------------------
# Assumes: row 0 matches row 0, row 1 matches row 1, etc.
df = pd.concat([df_phi.reset_index(drop=True),
                df_acc.reset_index(drop=True)],
               axis=1)

# Keep only the needed columns
df = df[["correct", "confidence"]].dropna()

# Confidence binning (safe version)
# ----------------------------
num_bins = 5
df["confidence_bin"] = pd.qcut(
    df["confidence"],
    num_bins,
    labels=False,
    duplicates="drop"
)

# See how many bins actually exist
unique_bins = df["confidence_bin"].nunique()
print(f"Number of unique confidence bins: {unique_bins}")

if unique_bins < 2:
    print("Not enough variability in confidence to bin reliably.")
    print("Computing correlation directly on raw confidence values instead.\n")

    corr, pval = pearsonr(df["confidence"], df["correct"])
    print(f"Correlation (raw confidence vs accuracy): {corr:.3f}")
    print(f"P-value: {pval:.4f}")
else:
    # Make bins start at 1 instead of 0
    df["confidence_bin"] = df["confidence_bin"] + 1

    grouped = df.groupby("confidence_bin", observed=True).agg({
        "confidence": "mean",
        "correct": "mean"
    }).reset_index()

    corr, pval = pearsonr(grouped["confidence"], grouped["correct"])

    print("=== Accuracy vs Confidence Analysis ===")
    print(grouped)
    print(f"\nCorrelation between confidence and accuracy: {corr:.3f}")
    print(f"P-value: {pval:.4f}")

    # Plot only if bins are valid
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot(grouped["confidence_bin"], grouped["correct"], marker="o", linewidth=2)
    plt.xlabel("Confidence Bin")
    plt.ylabel("Mean Accuracy")
    plt.title("Accuracy vs Confidence")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()