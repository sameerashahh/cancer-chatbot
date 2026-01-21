#!/usr/bin/env python3
"""
Analyze correlation between model performance and INVERSE complexity.
Replicates Figure 3 from FAMICOM paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

print("="*70)
print("INVERSE COMPLEXITY vs MODEL PERFORMANCE ANALYSIS")
print("="*70)
print("Using FIXED complexity scores")
print("="*70)

# Load data
print("\nLoading data...")
with open('model_results.json', 'r') as f:
    model_results = json.load(f)

with open('complexity_results_fixed.json', 'r') as f:
    complexity_results = json.load(f)

print(f"✓ Loaded {len(model_results)} model results")
print(f"✓ Loaded {len(complexity_results)} complexity scores")

# Extract data
accuracies = np.array([r['correct'] for r in model_results])
complexity_scores = np.array([c['guided'] for c in complexity_results])

# Calculate inverse complexity
# Handle division by zero: if complexity is 0, set inverse to a very small value
inverse_complexity = np.where(complexity_scores > 0, 
                               1.0 / complexity_scores, 
                               0.0)  # Or could use a very large number

print(f"\nData summary:")
print(f"  Total questions: {len(accuracies)}")
print(f"  Overall accuracy: {np.mean(accuracies)*100:.2f}%")
print(f"\nComplexity (guided) statistics:")
print(f"  Range: [{complexity_scores.min():.1f}, {complexity_scores.max():.1f}]")
print(f"  Mean: {complexity_scores.mean():.3f}")
print(f"  Unique values: {sorted(set(complexity_scores))}")
print(f"\nInverse Complexity statistics:")
print(f"  Range: [{inverse_complexity.min():.3f}, {inverse_complexity.max():.3f}]")
print(f"  Mean: {inverse_complexity.mean():.3f}")

# Filter out zero complexity questions for correlation analysis
valid_mask = complexity_scores > 0
valid_inverse_complexity = inverse_complexity[valid_mask]
valid_accuracies = accuracies[valid_mask]

print(f"\nValid questions (complexity > 0): {valid_mask.sum()}/{len(accuracies)}")

# Calculate Spearman correlation
correlation, p_value = spearmanr(valid_inverse_complexity, valid_accuracies)

print(f"\n{'='*70}")
print("SPEARMAN CORRELATION (Inverse Complexity vs Performance)")
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

# Create bins for visualization (like Figure 3 in the paper)
num_bins = 10
inv_complexity_bins = np.linspace(valid_inverse_complexity.min(), 
                                  valid_inverse_complexity.max(), 
                                  num_bins + 1)
bin_centers = []
bin_accuracies = []
bin_counts = []

print(f"\nBinning data into {num_bins} bins:")
for i in range(num_bins):
    if i == num_bins - 1:  # Last bin includes upper bound
        bin_mask = (valid_inverse_complexity >= inv_complexity_bins[i]) & \
                   (valid_inverse_complexity <= inv_complexity_bins[i + 1])
    else:
        bin_mask = (valid_inverse_complexity >= inv_complexity_bins[i]) & \
                   (valid_inverse_complexity < inv_complexity_bins[i + 1])
    
    if bin_mask.sum() > 0:
        bin_center = (inv_complexity_bins[i] + inv_complexity_bins[i + 1]) / 2
        bin_accuracy = valid_accuracies[bin_mask].mean()
        bin_centers.append(bin_center)
        bin_accuracies.append(bin_accuracy)
        bin_counts.append(bin_mask.sum())
        print(f"  Bin {i+1}: Inv.Complexity=[{inv_complexity_bins[i]:.3f}, {inv_complexity_bins[i+1]:.3f}] "
              f"→ Accuracy={bin_accuracy:.3f} (n={bin_mask.sum()})")

# Create plot like Figure 3 in the paper
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2.5, markersize=10, 
         color='#ff7f0e', markerfacecolor='#ff7f0e', markeredgewidth=2)

plt.xlabel('Inverse Complexity (1/Complexity)', fontsize=14, fontweight='bold')
plt.ylabel('Model Accuracy', fontsize=14, fontweight='bold')
plt.title(f'Model Performance vs. Inverse Complexity\n'
          f'Spearman ρ = {correlation:.3f}, p-value = {p_value:.2e}', 
          fontsize=15, fontweight='bold')

plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim([min(bin_accuracies) - 0.05, max(bin_accuracies) + 0.05])
plt.tight_layout()

# Save plot
output_file = 'complexity_performance_correlation.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to {output_file}")

# Save results to JSON
results = {
    "measure": "Inverse Complexity",
    "spearman_correlation": float(correlation),
    "p_value": float(p_value),
    "total_questions": len(accuracies),
    "valid_questions": int(valid_mask.sum()),
    "overall_accuracy": float(np.mean(accuracies)),
    "complexity_stats": {
        "mean": float(complexity_scores.mean()),
        "std": float(complexity_scores.std()),
        "min": float(complexity_scores.min()),
        "max": float(complexity_scores.max()),
        "unique_values": [float(x) for x in sorted(set(complexity_scores))]
    },
    "inverse_complexity_stats": {
        "mean": float(inverse_complexity.mean()),
        "std": float(inverse_complexity.std()),
        "min": float(inverse_complexity.min()),
        "max": float(inverse_complexity.max())
    },
    "binned_visualization": {
        "num_bins": num_bins,
        "bin_centers": [float(x) for x in bin_centers],
        "bin_accuracies": [float(x) for x in bin_accuracies],
        "bin_counts": [int(x) for x in bin_counts]
    }
}

with open('complexity_correlation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to complexity_correlation_results.json")

# Comparison with paper
print("\n" + "="*70)
print("COMPARISON WITH FAMICOM PAPER")
print("="*70)
print(f"Paper (Complexity):  Figure 3 shows positive correlation")
print(f"                     (better correlation than familiarity)")
print(f"Your result:         ρ = {correlation:.3f}, p = {p_value:.3e}")
print()
if correlation > 0.3:
    print("✅ Strong positive correlation - complexity predicts performance!")
elif correlation > 0.1:
    print("✅ Moderate positive correlation detected")
else:
    print("⚠️  Weak correlation")
print("="*70)

