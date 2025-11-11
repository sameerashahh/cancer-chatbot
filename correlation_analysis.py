import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler

# === Input files ===
FAMICOM_FILE = "outputs/famicom_combined_phi.json"
ACCURACY_FILE = "prompts_with_accuracy.json"

# === Load files ===
with open(FAMICOM_FILE, "r") as f:
    famicom_data = json.load(f)
with open(ACCURACY_FILE, "r") as f:
    accuracy_data = json.load(f)

df_famicom = pd.DataFrame(famicom_data)
df_acc = pd.DataFrame(accuracy_data)

min_len = min(len(df_famicom), len(df_acc))
df = pd.DataFrame({
    "famicom_score": df_famicom["famicom_score"].iloc[:min_len],
    "accuracy": df_acc["accuracy"].iloc[:min_len]
})

# === Normalize FAMICOM scores to 0–0.07 range (like paper) ===
scaler = MinMaxScaler(feature_range=(0, 0.07))
df["famicom_score_scaled"] = scaler.fit_transform(df[["famicom_score"]])

# === Bin scores for trend ===
bins = np.linspace(df["famicom_score_scaled"].min(), df["famicom_score_scaled"].max(), 10)
df["bin"] = pd.cut(df["famicom_score_scaled"], bins=bins, include_lowest=True)

grouped = df.groupby("bin").agg({
    "famicom_score_scaled": "mean",
    "accuracy": "mean"
}).reset_index()

# === Spearman correlation ===
corr, pval = spearmanr(df["famicom_score_scaled"], df["accuracy"])
print(f"Spearman correlation: {corr:.4f} (p = {pval:.6f})")

# === Plot ===
plt.figure(figsize=(6, 4))
plt.plot(grouped["famicom_score_scaled"], grouped["accuracy"], "-o", color="blue")
plt.xlabel("FAMICOM Score", fontsize=12)
plt.ylabel("Average Accuracy", fontsize=12)
plt.title(f"Correlation between Mistral Accuracy and FAMICOM (ρ = {corr:.2f})", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/famicom_trend_curve_scaled.png", dpi=300)
plt.show()
