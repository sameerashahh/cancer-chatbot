#!/usr/bin/env python3
"""
Analyze correlation between model performance and inverse complexity.
Replicates Figure 3 from FAMICOM paper.

Note: The paper uses INVERSE complexity (1/c) because:
- Higher complexity = harder question = lower performance
- Inverse complexity makes the correlation positive and more intuitive
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

print("="*70)
print("COMPLEXITY vs MODEL PERFORMANCE ANALYSIS")
print("="*70)

# Load data
print("\nLoading data...")
with open('model_results.json', 'r') as f:
    model_results = json.load(f)

with open('complexity_results (2).json', 'r') as f:
    complexity_results = json.load(f)

print(f"✓ Loaded {len(model_results)} model results")
print(f"✓ Loaded {len(complexity_results)} complexity scores")

# Extract data
accuracies = np.array([r['correct'] for r in model_results])
complexities = np.array([c['guided'] for c in complexity_results])

print(f"\nComplexity data summary:")
print(f"  Complexity range: [{complexities.min()}, {complexities.max()}]")
print(f"  Complexity mean: {complexities.mean():.2f}")
print(f"  Complexity distribution:")
for val in sorted(np.unique(complexities)):
    count = (complexities == val).sum()
    print(f"    Complexity {val}: {count} questions")

# Calculate inverse complexity (1/c)
# Handle complexity = 0 by adding small epsilon or filtering
# Paper approach: filter out complexity=0 or treat as very easy (high inverse)
inverse_complexities = np.zeros_like(complexities, dtype=float)
for i, c in enumerate(complexities):
    if c == 0:
        # Complexity 0 means very easy, so inverse complexity is very high
        # We'll set it to a large value (e.g., 10 or filter it out)
        inverse_complexities[i] = 10.0  # or could use np.nan and filter
    else:
        inverse_complexities[i] = 1.0 / c

print(f"\nInverse Complexity (1/c) summary:")
print(f"  Range: [{inverse_complexities.min():.3f}, {inverse_complexities.max():.3f}]")
print(f"  Mean: {inverse_complexities.mean():.3f}")

# Calculate Spearman correlation for INVERSE complexity
correlation, p_value = spearmanr(inverse_complexities, accuracies)

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
inv_complexity_bins = np.linspace(inverse_complexities.min(), inverse_complexities.max(), num_bins + 1)
bin_centers = []
bin_accuracies = []
bin_counts = []

print(f"\nBinning data into {num_bins} bins by inverse complexity:")
for i in range(num_bins):
    if i == num_bins - 1:  # Last bin includes upper bound
        bin_mask = (inverse_complexities >= inv_complexity_bins[i]) & (inverse_complexities <= inv_complexity_bins[i + 1])
    else:
        bin_mask = (inverse_complexities >= inv_complexity_bins[i]) & (inverse_complexities < inv_complexity_bins[i + 1])
    
    if bin_mask.sum() > 0:
        bin_center = (inv_complexity_bins[i] + inv_complexity_bins[i + 1]) / 2
        bin_accuracy = accuracies[bin_mask].mean()
        bin_centers.append(bin_center)
        bin_accuracies.append(bin_accuracy)
        bin_counts.append(bin_mask.sum())
        print(f"  Bin {i+1}: Inv.Complexity=[{inv_complexity_bins[i]:.3f}, {inv_complexity_bins[i+1]:.3f}] "
              f"→ Accuracy={bin_accuracy:.3f} (n={bin_mask.sum()})")

# Create plot like Figure 3 in the paper
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2.5, markersize=10, 
         color='#2ca02c', markerfacecolor='#2ca02c', markeredgewidth=2)

plt.xlabel('Inverse Complexity (1/c)', fontsize=14, fontweight='bold')
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

# Also analyze correlation with regular complexity (for comparison)
correlation_regular, p_value_regular = spearmanr(complexities, accuracies)
print(f"\n{'='*70}")
print("COMPARISON: Regular Complexity (not inverse)")
print(f"{'='*70}")
print(f"Correlation coefficient (ρ): {correlation_regular:.3f}")
print(f"P-value: {p_value_regular:.2e}")
print(f"Note: Should be negative (higher complexity → lower performance)")
print(f"{'='*70}")

# Save results to JSON
results = {
    "measure": "Inverse Complexity",
    "spearman_correlation": float(correlation),
    "p_value": float(p_value),
    "total_questions": len(accuracies),
    "overall_accuracy": float(np.mean(accuracies)),
    "complexity_stats": {
        "mean": float(complexities.mean()),
        "std": float(complexities.std()),
        "min": float(complexities.min()),
        "max": float(complexities.max())
    },
    "inverse_complexity_stats": {
        "mean": float(inverse_complexities.mean()),
        "std": float(inverse_complexities.std()),
        "min": float(inverse_complexities.min()),
        "max": float(inverse_complexities.max())
    },
    "regular_complexity_correlation": {
        "spearman_correlation": float(correlation_regular),
        "p_value": float(p_value_regular)
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
print(f"Paper (Inverse Complexity): ρ ≈ 0.5-0.6 (estimated from Figure 3)")
print(f"Your result:                ρ = {correlation:.3f}, p = {p_value:.3e}")
print()
if correlation > 0.3:
    print("✅ Positive correlation detected!")
    print("   Higher inverse complexity (easier questions) → Better performance")
else:
    print(f"⚠️  Weak correlation: {correlation:.3f}")
print("="*70)

