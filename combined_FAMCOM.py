import json
import math

# === Parameters ===
A = 1.0  # exponent for familiarity
B = 1.0  # exponent for complexity

# === Input Files ===
FAMILIARITY_FILE = "outputs/familiarity/phi3_transformers_64.json"
COMPLEXITY_FILE = "data/complexity_results_phi.json"
OUT_FILE = "outputs/famicom_combined_phi.json"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def combine_metrics(fam_data, comp_data):
    combined = []
    for fam, comp in zip(fam_data, comp_data):
        # --- FAMICOM formula components ---
        f_score = fam.get("token_similarity_top_ppl", 0)   # ✅ as per paper
        c_score = comp.get("guided", 0)                    # guided complexity

        # avoid division or zero errors
        f_score = max(f_score, 1e-6)
        c_score = max(c_score, 1e-6)

        # FAMICOM = f^a * c^(-b)
        famicom_score = (f_score ** A) * (c_score ** (-B))

        combined.append({
            "prompt": fam.get("text", comp.get("prompt", ""))[:300],
            "familiarity": round(f_score, 4),
            "complexity": round(c_score, 4),
            "famicom_score": round(famicom_score, 6)
        })
    return combined

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    fam_data = load_json(FAMILIARITY_FILE)
    comp_data = load_json(COMPLEXITY_FILE)
    combined = combine_metrics(fam_data, comp_data)
    save_json(combined, OUT_FILE)
    print(f"Combined FAMICOM scores saved to {OUT_FILE}")
    print(f"Example entry:\n{json.dumps(combined[0], indent=2)}")

if __name__ == "__main__":
    main()
