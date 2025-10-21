#!/usr/bin/env python3
"""
Analyze correlation between model performance and familiarity.
Replicates Figure 2 from FAMICOM paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

print("="*70)
print("FAMILIARITY vs MODEL PERFORMANCE ANALYSIS")
print("="*70)

# Load data
print("\nLoading data...")
with open('model_results.json', 'r') as f:
    model_results = json.load(f)

with open('familiarity_scores.json', 'r') as f:
    familiarity_scores = json.load(f)

print(f"✓ Loaded {len(model_results)} model results")
print(f"✓ Loaded {len(familiarity_scores)} familiarity scores")

# Extract data
accuracies = np.array([r['correct'] for r in model_results])
familiarities = np.array([f['familiarity_score'] for f in familiarity_scores])

print(f"\nData summary:")
print(f"  Total questions: {len(accuracies)}")
print(f"  Overall accuracy: {np.mean(accuracies)*100:.2f}%")
print(f"  Familiarity range: [{familiarities.min():.4f}, {familiarities.max():.4f}]")
print(f"  Familiarity mean: {familiarities.mean():.4f}")

# Calculate Spearman correlation
correlation, p_value = spearmanr(familiarities, accuracies)

print(f"\n{'='*70}")
print("SPEARMAN CORRELATION")
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

# Create bins for visualization (like Figure 2 in the paper)
num_bins = 10
familiarity_bins = np.linspace(familiarities.min(), familiarities.max(), num_bins + 1)
bin_centers = []
bin_accuracies = []
bin_counts = []

print(f"\nBinning data into {num_bins} bins:")
for i in range(num_bins):
    if i == num_bins - 1:  # Last bin includes upper bound
        bin_mask = (familiarities >= familiarity_bins[i]) & (familiarities <= familiarity_bins[i + 1])
    else:
        bin_mask = (familiarities >= familiarity_bins[i]) & (familiarities < familiarity_bins[i + 1])
    
    if bin_mask.sum() > 0:
        bin_center = (familiarity_bins[i] + familiarity_bins[i + 1]) / 2
        bin_accuracy = accuracies[bin_mask].mean()
        bin_centers.append(bin_center)
        bin_accuracies.append(bin_accuracy)
        bin_counts.append(bin_mask.sum())
        print(f"  Bin {i+1}: Familiarity=[{familiarity_bins[i]:.3f}, {familiarity_bins[i+1]:.3f}] "
              f"→ Accuracy={bin_accuracy:.3f} (n={bin_mask.sum()})")

# Create plot like Figure 2 in the paper
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2.5, markersize=10, 
         color='#1f77b4', markerfacecolor='#1f77b4', markeredgewidth=2)

plt.xlabel('Familiarity', fontsize=14, fontweight='bold')
plt.ylabel('Model Accuracy', fontsize=14, fontweight='bold')
plt.title(f'Model Performance vs. Familiarity\n'
          f'Spearman ρ = {correlation:.3f}, p-value = {p_value:.2e}', 
          fontsize=15, fontweight='bold')

plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim([min(bin_accuracies) - 0.05, max(bin_accuracies) + 0.05])
plt.tight_layout()

# Save plot
output_file = 'familiarity_performance_correlation.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to {output_file}")

# Save results to JSON
results = {
    "measure": "Familiarity",
    "spearman_correlation": float(correlation),
    "p_value": float(p_value),
    "total_questions": len(accuracies),
    "overall_accuracy": float(np.mean(accuracies)),
    "familiarity_stats": {
        "mean": float(familiarities.mean()),
        "std": float(familiarities.std()),
        "min": float(familiarities.min()),
        "max": float(familiarities.max())
    },
    "binned_visualization": {
        "num_bins": num_bins,
        "bin_centers": [float(x) for x in bin_centers],
        "bin_accuracies": [float(x) for x in bin_accuracies],
        "bin_counts": [int(x) for x in bin_counts]
    }
}

with open('familiarity_correlation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to familiarity_correlation_results.json")

# Comparison with paper
print("\n" + "="*70)
print("COMPARISON WITH FAMICOM PAPER")
print("="*70)
print(f"Paper (Familiarity):  ρ ≈ 0.426, p = 0.002")
print(f"Your result:          ρ = {correlation:.3f}, p = {p_value:.3e}")
print()
if abs(correlation - 0.426) < 0.1:
    print("✅ Your result is consistent with the paper!")
else:
    print(f"⚠️  Difference from paper: {abs(correlation - 0.426):.3f}")
print("="*70)

