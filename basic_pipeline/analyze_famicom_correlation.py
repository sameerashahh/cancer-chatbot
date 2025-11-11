#!/usr/bin/env python3
"""
Analyze correlation between FAMICOM score and model performance.
FAMICOM = familiarity^a × complexity^(-b) where a=b=1
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

print("="*70)
print("FAMICOM vs MODEL PERFORMANCE ANALYSIS")
print("="*70)

# Load data
print("\nLoading data...")
with open('model_results.json', 'r') as f:
    model_results = json.load(f)

with open('famicom_scores.json', 'r') as f:
    famicom_data = json.load(f)

print(f"✓ Loaded {len(model_results)} model results")
print(f"✓ Loaded {len(famicom_data)} FAMICOM scores")

# Extract data from pre-calculated FAMICOM file
accuracies = np.array([r['correct'] for r in model_results])
familiarities = np.array([f['familiarity'] for f in famicom_data])
complexities = np.array([f['complexity'] for f in famicom_data])
famicom_scores = np.array([f['famicom_score'] for f in famicom_data])

print(f"\nData summary:")
print(f"  Total questions: {len(accuracies)}")
print(f"  Overall accuracy: {np.mean(accuracies)*100:.2f}%")
print(f"\nFamiliarity:")
print(f"  Range: [{familiarities.min():.4f}, {familiarities.max():.4f}]")
print(f"  Mean: {familiarities.mean():.4f}")
print(f"\nComplexity:")
print(f"  Range: [{complexities.min():.1f}, {complexities.max():.1f}]")
print(f"  Mean: {complexities.mean():.2f}")
print(f"\nFAMICOM Score:")
print(f"  Range: [{famicom_scores.min():.4f}, {famicom_scores.max():.4f}]")
print(f"  Mean: {famicom_scores.mean():.4f}")

# Calculate Spearman correlation
correlation, p_value = spearmanr(famicom_scores, accuracies)

print(f"\n{'='*70}")
print("SPEARMAN CORRELATION (FAMICOM vs Performance)")
print(f"{'='*70}")
print(f"Correlation coefficient (ρ): {correlation:.3f}")
print(f"P-value: {p_value:.2e}")
if p_value < 0.001:
    print(f"Significance: *** (p < 0.001)")
elif p_value < 0.01:
    print(f"Significance: ** (p < 0.01)")
elif p_value < 0.05:
    print(f"Significance: * (p < 0.05)")
else:
    print(f"Significance: not significant (p >= 0.05)")
print(f"{'='*70}")

# Create bins for visualization
num_bins = 10
famicom_bins = np.linspace(famicom_scores.min(), famicom_scores.max(), num_bins + 1)
bin_centers = []
bin_accuracies = []
bin_counts = []

print(f"\nBinning data into {num_bins} bins:")
for i in range(num_bins):
    if i == num_bins - 1:  # Last bin includes upper bound
        bin_mask = (famicom_scores >= famicom_bins[i]) & (famicom_scores <= famicom_bins[i + 1])
    else:
        bin_mask = (famicom_scores >= famicom_bins[i]) & (famicom_scores < famicom_bins[i + 1])
    
    if bin_mask.sum() > 0:
        bin_center = (famicom_bins[i] + famicom_bins[i + 1]) / 2
        bin_accuracy = accuracies[bin_mask].mean()
        bin_centers.append(bin_center)
        bin_accuracies.append(bin_accuracy)
        bin_counts.append(bin_mask.sum())
        print(f"  Bin {i+1}: FAMICOM=[{famicom_bins[i]:.3f}, {famicom_bins[i+1]:.3f}] "
              f"→ Accuracy={bin_accuracy:.3f} (n={bin_mask.sum()})")

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2.5, markersize=10, 
         color='#2ca02c', markerfacecolor='#2ca02c', markeredgewidth=2)

plt.xlabel('FAMICOM Score (Familiarity / Complexity)', fontsize=14, fontweight='bold')
plt.ylabel('Model Accuracy', fontsize=14, fontweight='bold')
plt.title(f'Model Performance vs. FAMICOM Score\n'
          f'Spearman ρ = {correlation:.3f}, p-value = {p_value:.2e}', 
          fontsize=15, fontweight='bold')

plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim([min(bin_accuracies) - 0.05, max(bin_accuracies) + 0.05])
plt.tight_layout()

# Save plot
output_file = 'famicom_performance_correlation.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to {output_file}")

# Save results to JSON
results = {
    "measure": "FAMICOM (Familiarity / Complexity)",
    "formula": "f^1 × c^(-1)",
    "spearman_correlation": float(correlation),
    "p_value": float(p_value),
    "total_questions": len(accuracies),
    "overall_accuracy": float(np.mean(accuracies)),
    "famicom_stats": {
        "mean": float(famicom_scores.mean()),
        "std": float(famicom_scores.std()),
        "min": float(famicom_scores.min()),
        "max": float(famicom_scores.max())
    },
    "binned_visualization": {
        "num_bins": num_bins,
        "bin_centers": [float(x) for x in bin_centers],
        "bin_accuracies": [float(x) for x in bin_accuracies],
        "bin_counts": [int(x) for x in bin_counts]
    }
}

with open('famicom_correlation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to famicom_correlation_results.json")

# Comparison with all metrics
print("\n" + "="*70)
print("COMPARISON OF ALL METRICS")
print("="*70)

corr_fam, _ = spearmanr(familiarities, accuracies)
corr_comp, _ = spearmanr(complexities, accuracies)

print(f"\nYour Results (Phi-3):")
print(f"  Familiarity:  ρ = {corr_fam:.3f}")
print(f"  Complexity:   ρ = {corr_comp:.3f}")
print(f"  FAMICOM:      ρ = {correlation:.3f}")
print()

best_corr = max(abs(corr_fam), abs(corr_comp), abs(correlation))
if abs(correlation) == best_corr:
    print("✅ FAMICOM is the BEST predictor!")
elif abs(correlation) > 0.9 * best_corr:
    print("✓ FAMICOM is competitive with the best predictor")
else:
    if abs(corr_fam) == best_corr:
        print("⚠️  Familiarity alone is better than FAMICOM")
    else:
        print("⚠️  Complexity alone is better than FAMICOM")

print()
print("Paper expectations:")
print(f"  FAMICOM: ρ > 0.6 (should be BEST predictor)")
print(f"  Your FAMICOM: ρ = {correlation:.3f}")
print()

if correlation > corr_fam and correlation > corr_comp:
    print("✅ FAMICOM combines both factors successfully!")
else:
    print("Note: For Phi-3, FAMICOM doesn't outperform individual metrics.")
    print("This suggests Phi-3 behaves differently than the paper's Mistral model.")

print("="*70)

