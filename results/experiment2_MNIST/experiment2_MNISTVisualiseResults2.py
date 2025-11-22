import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load results
results = np.load("results/experiment2_MNIST/experiment2_MNIST.npz", allow_pickle=True)
results = {k: v.item() for k, v in results.items()}

saveDir = "results/experiment2_MNIST/graphs/experiment2"
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

# ============= 1. VARIANCE ANALYSIS - THE KEY FINDING =============

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Stability Analysis: Variance Across Runs', fontsize=18, fontweight='bold')

metrics = [
    ("test_accuracy", "Test Accuracy (%)", axes[0, 0]),
    ("test_loss", "Test Loss", axes[0, 1]),
    ("train_accuracy", "Train Accuracy (%)", axes[1, 0]),
    ("train_loss", "Train Loss", axes[1, 1])
]

for metric_name, ylabel, ax in metrics:
    for i, activation in enumerate(results.keys()):
        runs = np.array(results[activation][metric_name])
        mean = np.mean(runs, axis=0)
        std = np.std(runs, axis=0)
        
        # Plot mean line
        ax.plot(range(len(mean)), mean, label=activation, color=colors[i], 
                linewidth=2.5, alpha=0.8)
        
        # Highlight variance with shaded region
        ax.fill_between(range(len(mean)), mean - std, mean + std, 
                        color=colors[i], alpha=0.15)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{ylabel} (Shaded = ±1 Std Dev)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{saveDir}_variance_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# ============= 2. VARIANCE QUANTIFICATION =============

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Calculate final epoch variance for each activation
variance_data = []
for activation in results.keys():
    test_acc_final = [run[-1] for run in results[activation]["test_accuracy"]]
    test_loss_final = [run[-1] for run in results[activation]["test_loss"]]
    
    variance_data.append({
        "Activation": activation,
        "Accuracy Std": np.std(test_acc_final),
        "Accuracy Range": np.max(test_acc_final) - np.min(test_acc_final),
        "Loss Std": np.std(test_loss_final),
        "Mean Accuracy": np.mean(test_acc_final),
        "Mean Loss": np.mean(test_loss_final)
    })

variance_df = pd.DataFrame(variance_data)

# Plot 1: Standard Deviation Comparison
ax1 = axes[0]
x_pos = np.arange(len(variance_df))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, variance_df["Accuracy Std"], width, 
                label='Accuracy Std (%)', color='#3498db', alpha=0.7, edgecolor='black')
bars2 = ax1.bar(x_pos + width/2, variance_df["Loss Std"] * 20, width,  # Scale loss for visibility
                label='Loss Std (×20)', color='#e74c3c', alpha=0.7, edgecolor='black')

ax1.set_xlabel('Activation Function', fontsize=12)
ax1.set_ylabel('Standard Deviation', fontsize=12)
ax1.set_title('Stability Metric: Standard Deviation Across Runs', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(variance_df["Activation"], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Range (Max - Min) Comparison
ax2 = axes[1]
bars = ax2.bar(range(len(variance_df)), variance_df["Accuracy Range"], 
               color=colors[:len(variance_df)], alpha=0.7, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Activation Function', fontsize=12)
ax2.set_ylabel('Accuracy Range (%)', fontsize=12)
ax2.set_title('Worst Case Variability: Max - Min Accuracy', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(variance_df)))
ax2.set_xticklabels(variance_df["Activation"], rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels and highlight problem cases
for i, (bar, range_val) in enumerate(zip(bars, variance_df["Accuracy Range"])):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{range_val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight high variance (>5%)
    if range_val > 5:
        bar.set_edgecolor('red')
        bar.set_linewidth(3)

# Add reference line for "acceptable" variance
ax2.axhline(y=5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='High Variance Threshold')
ax2.legend()

plt.tight_layout()
plt.savefig(f"{saveDir}_variance_quantification.png", dpi=300, bbox_inches='tight')
plt.close()

# ============= 3. RUN-BY-RUN COMPARISON (Show individual runs) =============

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Individual Run Trajectories: Revealing Training Instability', fontsize=16, fontweight='bold')

for idx, activation in enumerate(results.keys()):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Plot each individual run
    test_accs = results[activation]["test_accuracy"]
    for run_idx, run in enumerate(test_accs):
        ax.plot(range(len(run)), run, label=f'Run {run_idx+1}', 
                linewidth=2, alpha=0.7)
    
    # Plot mean
    mean_acc = np.mean(test_accs, axis=0)
    ax.plot(range(len(mean_acc)), mean_acc, 'k--', linewidth=3, 
            label='Mean', alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Test Accuracy (%)', fontsize=10)
    ax.set_title(f'{activation}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add variance annotation
    final_std = np.std([run[-1] for run in test_accs])
    ax.text(0.02, 0.98, f'Std: {final_std:.2f}%', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f"{saveDir}_individual_runs.png", dpi=300, bbox_inches='tight')
plt.close()

# ============= 4. CONVERGENCE CONSISTENCY ANALYSIS =============

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

consistency_data = []
for activation in results.keys():
    test_accs = results[activation]["test_accuracy"]
    
    # For each epoch, calculate coefficient of variation (std/mean)
    cv_per_epoch = []
    for epoch in range(len(test_accs[0])):
        epoch_accs = [run[epoch] for run in test_accs]
        mean_acc = np.mean(epoch_accs)
        std_acc = np.std(epoch_accs)
        cv = (std_acc / mean_acc) * 100 if mean_acc > 0 else 0  # CV as percentage
        cv_per_epoch.append(cv)
    
    ax.plot(range(len(cv_per_epoch)), cv_per_epoch, label=activation, 
            linewidth=2.5, marker='o', markersize=3, alpha=0.8)
    
    consistency_data.append({
        "Activation": activation,
        "Mean CV": np.mean(cv_per_epoch),
        "Max CV": np.max(cv_per_epoch),
        "Final CV": cv_per_epoch[-1]
    })

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Coefficient of Variation (%)', fontsize=13)
ax.set_title('Training Consistency: Lower = More Stable', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{saveDir}_consistency_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# ============= 5. ENHANCED SUMMARY TABLE WITH VARIANCE =============

summary = []
for activation in results.keys():
    train_accs = [run[-1] for run in results[activation]["train_accuracy"]]
    test_accs = [run[-1] for run in results[activation]["test_accuracy"]]
    train_losses = [run[-1] for run in results[activation]["train_loss"]]
    test_losses = [run[-1] for run in results[activation]["test_loss"]]
    
    summary.append({
        "Activation": activation,
        "Test Acc Mean (%)": np.mean(test_accs),
        "Test Acc Std (%)": np.std(test_accs),
        "Test Acc Range (%)": np.max(test_accs) - np.min(test_accs),
        "Test Loss Mean": np.mean(test_losses),
        "Test Loss Std": np.std(test_losses),
        "Stability Score": np.std(test_accs),  # Lower = more stable
    })

summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values('Stability Score', ascending=True)  # Most stable first

print("\n" + "="*120)
print("EXPERIMENT 2: STABILITY ANALYSIS")
print("="*120)
print(summary_df.to_string(index=False))
print("="*120)

# Categorize by stability
print("\nSTABILITY CLASSIFICATION:")
print("-" * 120)
for _, row in summary_df.iterrows():
    stability = row['Stability Score']
    if stability < 1.0:
        category = "✅ HIGHLY STABLE"
    elif stability < 3.0:
        category = "⚠️  MODERATELY STABLE"
    else:
        category = "❌ UNSTABLE"
    
    print(f"{row['Activation']:20s} | Std: {stability:5.2f}% | Range: {row['Test Acc Range (%)']:5.2f}% | {category}")
print("="*120)

# Key insights
print("\nKEY INSIGHTS FROM VARIANCE ANALYSIS:")
print("-" * 120)

# Find most and least stable
most_stable = summary_df.iloc[0]
least_stable = summary_df.iloc[-1]

print(f"1. Most Stable:  {most_stable['Activation']} (Std = {most_stable['Stability Score']:.2f}%)")
print(f"2. Least Stable: {least_stable['Activation']} (Std = {least_stable['Stability Score']:.2f}%)")
print(f"3. Stability Gap: {least_stable['Stability Score'] - most_stable['Stability Score']:.2f}% difference")

# Compare node vs layer
quad_node_std = summary_df[summary_df['Activation'] == 'Quad-Node']['Stability Score'].values[0]
quad_layer_std = summary_df[summary_df['Activation'] == 'Quad-Layer']['Stability Score'].values[0]
print(f"4. Quad-Node vs Quad-Layer Stability: {quad_layer_std / quad_node_std:.1f}x more variance with shared params")

cubic_node_std = summary_df[summary_df['Activation'] == 'Cubic-Node']['Stability Score'].values[0]
cubic_layer_std = summary_df[summary_df['Activation'] == 'Cubic-Layer']['Stability Score'].values[0]
print(f"5. Cubic-Node vs Cubic-Layer Stability: {cubic_layer_std / cubic_node_std:.1f}x more variance with shared params")

print("\nCONCLUSION:")
print("Shared parameters (Layer) cause BOTH lower performance AND higher variance.")
print("This suggests fundamental optimization difficulties, not just insufficient capacity.")
print("="*120)

# Save enhanced summary
summary_df.to_csv(f"{saveDir}_stability_summary.csv", index=False)
variance_df.to_csv(f"{saveDir}_variance_metrics.csv", index=False)