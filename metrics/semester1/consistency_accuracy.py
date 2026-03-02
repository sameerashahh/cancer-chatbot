import json
import pandas as pd

# ----------------------------
# Load evaluation results (self-consistency outputs)
# ----------------------------
with open("self_consistency_results.json") as f:
    sc_data = json.load(f)

with open("phi_evaluation_results.json") as f:
    true_data = json.load(f)

# ----------------------------
# Convert to DataFrames
# ----------------------------
df_sc = pd.DataFrame(sc_data)
df_true = pd.DataFrame(true_data)

# Merge on question text (or index if needed)
df = pd.merge(df_sc, df_true, on="question", how="inner", suffixes=("_sc","_true"))

# ----------------------------
# Compute accuracy
# ----------------------------
df["correct"] = (df["majority_label"] == df["true_answer"]).astype(int)
accuracy = df["correct"].mean()

print("=== Self-Consistency Accuracy ===")
print(f"Total prompts evaluated: {len(df)}")
print(f"Number correct: {df['correct'].sum()}")
print(f"Accuracy: {accuracy:.3f}")

# ----------------------------
# Optional: Save per-prompt results
# ----------------------------
df.to_json("self_consistency_vs_true_answer.json", orient="records", indent=2)
