#!/usr/bin/env python3
"""
Calculate FAMICOM scores using the FAMICOM paper formula:
FAMICOM = f^a * c^(-b)
where f = familiarity, c = complexity, a = b = 1.0
"""

import json
import numpy as np
from scipy.stats import spearmanr

# === Parameters (from paper) ===
A = 1.0  # exponent for familiarity
B = 1.0  # exponent for complexity

# === Input Files ===
FAMILIARITY_FILE = "familiarity_scores.json"
COMPLEXITY_FILE = "complexity_results_fixed.json"  # FIXED: Use corrected complexity scores
MODEL_RESULTS_FILE = "model_results.json"
OUT_FILE = "famicom_scores.json"

def load_json(path):
    """Load JSON file"""
    with open(path, "r") as f:
        return json.load(f)

def calculate_famicom_scores(fam_data, comp_data, model_data):
    """
    Calculate FAMICOM scores for each question.
    
    FAMICOM = f^a * c^(-b)
    - f = familiarity score (0 to 1)
    - c = complexity score (1 to 5+)
    - Higher FAMICOM = easier question (high familiarity, low complexity)
    """
    combined = []
    
    for i, (fam, comp, model) in enumerate(zip(fam_data, comp_data, model_data)):
        # Extract familiarity score
        f_score = fam.get("familiarity_score", 0)
        
        # Extract complexity score
        c_score = comp.get("guided", 0)
        
        # Avoid division by zero or invalid values
        f_score = max(f_score, 1e-6)
        c_score = max(c_score, 1e-6)
        
        # FAMICOM formula: f^a * c^(-b)
        # Higher familiarity → higher FAMICOM
        # Higher complexity → lower FAMICOM (due to negative exponent)
        famicom_score = (f_score ** A) * (c_score ** (-B))
        
        combined.append({
            "question_id": i,
            "dataset": comp.get("dataset", ""),
            "subtask": comp.get("subtask", ""),
            "question": comp.get("question", "")[:200],
            "true_answer": model.get("true_answer", ""),
            "model_answer": model.get("model_answer", ""),
            "correct": model.get("correct", 0),
            "familiarity": round(f_score, 4),
            "complexity": round(c_score, 4),
            "famicom_score": round(famicom_score, 6)
        })
    
    return combined

def save_json(data, path):
    """Save data to JSON file"""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    print("="*70)
    print("FAMICOM SCORE CALCULATION")
    print("="*70)
    
    # Load data
    print(f"\nLoading data...")
    print(f"  Familiarity: {FAMILIARITY_FILE}")
    print(f"  Complexity: {COMPLEXITY_FILE}")
    print(f"  Model Results: {MODEL_RESULTS_FILE}")
    
    fam_data = load_json(FAMILIARITY_FILE)
    comp_data = load_json(COMPLEXITY_FILE)
    model_data = load_json(MODEL_RESULTS_FILE)
    
    print(f"\n✓ Loaded {len(fam_data)} familiarity scores")
    print(f"✓ Loaded {len(comp_data)} complexity scores")
    print(f"✓ Loaded {len(model_data)} model results")
    
    # Verify lengths match
    if not (len(fam_data) == len(comp_data) == len(model_data)):
        print("\n⚠️  WARNING: File lengths don't match!")
        print(f"   Familiarity: {len(fam_data)}")
        print(f"   Complexity: {len(comp_data)}")
        print(f"   Model: {len(model_data)}")
        print("   Using minimum length...")
        min_len = min(len(fam_data), len(comp_data), len(model_data))
        fam_data = fam_data[:min_len]
        comp_data = comp_data[:min_len]
        model_data = model_data[:min_len]
    
    # Calculate FAMICOM scores
    print(f"\nCalculating FAMICOM scores...")
    print(f"  Formula: FAMICOM = familiarity^{A} × complexity^(-{B})")
    
    combined = calculate_famicom_scores(fam_data, comp_data, model_data)
    
    # Save results
    save_json(combined, OUT_FILE)
    print(f"\n✅ FAMICOM scores saved to {OUT_FILE}")
    
    # Print statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    famicom_scores = np.array([c['famicom_score'] for c in combined])
    familiarities = np.array([c['familiarity'] for c in combined])
    complexities = np.array([c['complexity'] for c in combined])
    accuracies = np.array([c['correct'] for c in combined])
    
    print(f"\nFamiliarity:")
    print(f"  Mean: {familiarities.mean():.4f}")
    print(f"  Range: [{familiarities.min():.4f}, {familiarities.max():.4f}]")
    
    print(f"\nComplexity:")
    print(f"  Mean: {complexities.mean():.4f}")
    print(f"  Range: [{complexities.min():.1f}, {complexities.max():.1f}]")
    
    print(f"\nFAMICOM Score:")
    print(f"  Mean: {famicom_scores.mean():.6f}")
    print(f"  Range: [{famicom_scores.min():.6f}, {famicom_scores.max():.6f}]")
    
    print(f"\nModel Accuracy: {accuracies.mean()*100:.2f}%")
    
    # Calculate correlations
    print("\n" + "="*70)
    print("CORRELATIONS WITH MODEL ACCURACY")
    print("="*70)
    
    # Filter out zero complexity for cleaner correlations
    valid_mask = complexities > 0
    
    corr_fam, p_fam = spearmanr(familiarities[valid_mask], accuracies[valid_mask])
    corr_comp, p_comp = spearmanr(complexities[valid_mask], accuracies[valid_mask])
    corr_famicom, p_famicom = spearmanr(famicom_scores[valid_mask], accuracies[valid_mask])
    
    print(f"\nFamiliarity vs Accuracy:")
    print(f"  Spearman ρ = {corr_fam:.3f}, p = {p_fam:.3e}")
    
    print(f"\nComplexity vs Accuracy:")
    print(f"  Spearman ρ = {corr_comp:.3f}, p = {p_comp:.3e}")
    
    print(f"\nFAMICOM vs Accuracy:")
    print(f"  Spearman ρ = {corr_famicom:.3f}, p = {p_famicom:.3e}")
    
    print("\n" + "="*70)
    print("COMPARISON WITH PAPER")
    print("="*70)
    print(f"Paper expectations:")
    print(f"  Familiarity: ρ ≈ 0.426")
    print(f"  Complexity: ρ ≈ 0.6")
    print(f"  FAMICOM: ρ > 0.6 (BEST predictor)")
    print()
    print(f"Your results:")
    print(f"  Familiarity: ρ = {corr_fam:.3f}")
    print(f"  Complexity: ρ = {corr_comp:.3f}")
    print(f"  FAMICOM: ρ = {corr_famicom:.3f}")
    print()
    
    if corr_famicom > corr_fam and corr_famicom > corr_comp:
        print("✅ FAMICOM is the best predictor!")
    elif corr_famicom > max(corr_fam, corr_comp) * 0.8:
        print("✓ FAMICOM is competitive")
    else:
        print("⚠️  FAMICOM is not the strongest predictor")
        print("   This may be due to complexity score issues")
    
    # Example entries
    print("\n" + "="*70)
    print("EXAMPLE ENTRIES")
    print("="*70)
    
    # Show highest and lowest FAMICOM scores
    sorted_indices = np.argsort(famicom_scores)
    
    print("\nHighest FAMICOM (easiest questions):")
    for idx in sorted_indices[-3:]:
        entry = combined[idx]
        print(f"\n  FAMICOM: {entry['famicom_score']:.4f} (F={entry['familiarity']:.3f}, C={entry['complexity']:.1f})")
        print(f"  Question: {entry['question'][:100]}...")
        print(f"  Correct: {'✓' if entry['correct'] else '✗'}")
    
    print("\n\nLowest FAMICOM (hardest questions):")
    for idx in sorted_indices[:3]:
        entry = combined[idx]
        print(f"\n  FAMICOM: {entry['famicom_score']:.4f} (F={entry['familiarity']:.3f}, C={entry['complexity']:.1f})")
        print(f"  Question: {entry['question'][:100]}...")
        print(f"  Correct: {'✓' if entry['correct'] else '✗'}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

