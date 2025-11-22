import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load results
results = np.load("results/experiment3_SyntheticData/experiment3_SyntheticData.npz", allow_pickle=True)
results = {k: v.item() for k, v in results.items()}  # assumes each v is a 0-d object array containing a dict

saveDir = "results/experiment3_SyntheticData/graphs/experiment3"

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'ReLU': '#e74c3c', 'Sigmoid': '#3498db', 'Quad-Node': '#2ecc71', 'Cubic-Node': '#f39c12'}

tasks = ["linear", "polynomial", "discontinuous", "oscillatory"]
task_labels = ["Linear", "Polynomial", "Discontinuous", "Oscillatory"]
activations = ["ReLU", "Sigmoid", "Quad-Node", "Cubic-Node"]

# ============================================================
# 1. HEATMAP: Final Test Loss by Task × Activation
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
heatmap_data = np.zeros((len(tasks), len(activations)))

for i, task in enumerate(tasks):
    for j, activation in enumerate(activations):
        final_losses = [run[-1] for run in results[task][activation]["test_loss"]]
        heatmap_data[i, j] = np.mean(final_losses)

sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn_r',
            xticklabels=activations, yticklabels=task_labels,
            cbar_kws={'label': 'Final Test Loss (MSE)'}, ax=ax,
            vmin=0, vmax=np.max(heatmap_data))

ax.set_title('Task Performance Heatmap: Final Test Loss (Lower = Better)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Activation Function', fontsize=13)
ax.set_ylabel('Task Type', fontsize=13)

plt.tight_layout()
plt.savefig(f"{saveDir}_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()


# ============================================================
# 2. LEARNING CURVES: Test Loss Over Epochs (All Tasks)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for idx, (task, task_label) in enumerate(zip(tasks, task_labels)):
    ax = axes[idx]
    
    for activation in activations:
        runs = np.array(results[task][activation]["test_loss"])
        mean = np.mean(runs, axis=0)
        std = np.std(runs, axis=0)
        
        ax.plot(range(len(mean)), mean, label=activation, color=colors[activation],
                linewidth=2.5, alpha=0.8)
        ax.fill_between(range(len(mean)), mean - std, mean + std,
                        color=colors[activation], alpha=0.15)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Loss (MSE)', fontsize=12)
    ax.set_title(f'{task_label} Task', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization

plt.suptitle('Learning Curves: Test Loss Over Epochs (Log Scale)',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(f"{saveDir}_test_loss.png", dpi=300, bbox_inches='tight')
plt.close()


# ============================================================
# 3. LEARNING CURVES: Train Loss Over Epochs (All Tasks)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for idx, (task, task_label) in enumerate(zip(tasks, task_labels)):
    ax = axes[idx]
    
    for activation in activations:
        runs = np.array(results[task][activation]["train_loss"])
        mean = np.mean(runs, axis=0)
        std = np.std(runs, axis=0)
        
        ax.plot(range(len(mean)), mean, label=activation, color=colors[activation],
                linewidth=2.5, alpha=0.8)
        ax.fill_between(range(len(mean)), mean - std, mean + std,
                        color=colors[activation], alpha=0.15)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train Loss (MSE)', fontsize=12)
    ax.set_title(f'{task_label} Task', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.suptitle('Learning Curves: Train Loss Over Epochs (Log Scale)',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(f"{saveDir}_train_loss.png", dpi=300, bbox_inches='tight')
plt.close()
