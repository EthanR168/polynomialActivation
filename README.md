# Learnable Polynomial Activation Functions

Kolmogorov-Arnold Network (KAN) is a new type of neural network inspired by the Kolmogorov-Arnold representation theorem and was introduced in the paper "KAN: Kolmogorov-Arnold Network" on April 30 2024. The idea which made KAN different from traditional Multi Layer Perceptrons (MLPs) was the idea of using b splines as learnable activation functions on edges between nodes instead of fixed activation functions on nodes.

I found the idea of learnable activation functions interesting, however b splines are very complicated and so instead I swapped b splines for polynomial functions with learnable coefficients.

---

## Key Question

Can learnable polynomial activations (ax^2 + bx + c and cubic variant) work as alternatives to ReLU?

---

# Overview

Three experiments were conducted:

1. Normalisation strategies for polynomial activations  
2. MNIST classification benchmark  
3. Synthetic mathematical regression tasks

This README summarises all three, showing where polynomial activations succeed and where they fail.

Each experiment has their individual summaries:
-   [Experiment 1: Normalisation](/results/experiment1_Normalisation/summary.md)
-   [Experiment 2: MNIST](/results/experiment2_MNIST/summary.md)
-   [Experiment 3: Synthetic Datasets](/results/experiment3_SyntheticData/summary.md)

---

# Experiment 1: Normalisation

Polynomial activations grow rapidly and without control they cause exploding outputs and overflow errors.  
This experiment compares BatchNorm, LayerNorm, Tanh, clipping, and no normalisation.

### Plots

![Train Accuracy per Epoch](/results/experiment1_Normalisation/graphs/experiment1_train_accuracy.png)
![Test Accuracy per Epoch](/results/experiment1_Normalisation/graphs/experiment1_test_accuracy.png)
![Train Loss per Epoch](/results/experiment1_Normalisation/graphs/experiment1_train_loss.png)
![Test Loss per Epoch](/results/experiment1_Normalisation/graphs/experiment1_test_loss.png)

### Results

| Normalisation | Train Acc | Test Acc | Train Loss | Test Loss | Converged |
|---------------|-----------|----------|------------|-----------|-----------|
| BatchNorm     | 81.75     | 84.74    | 0.584      | 0.499     | True      |
| LayerNorm     | 83.77     | 83.84    | 0.527      | 0.517     | True      |
| Tanh          | 74.05     | 74.42    | 0.815      | 0.798     | True      |
| clip          | 70.41     | 70.46    | 0.956      | 0.949     | True      |
| No Norm       | 9.87      | 9.80     | NaN        | NaN       | False     |

### Key Findings

- Normalisation is required as without it polynomial activations explode and the model does not converge or learn at all.
- Tanh and clipping prevent explosion but do not stabilise distributions across layers.  
- LayerNorm is the most stable normalisation strategy.

---

# Experiment 2: MNIST Benchmark

This experiment compares:

- ReLU  
- Sigmoid  
- Polynomial activations with per-neuron coefficients  
- Polynomial activations with shared layer coefficients  
- Quadratic vs cubic polynomials  

### Plots

![Test Accuracy per Epoch](/results/experiment2_MNIST/graphs/experiment2_test_accuracy.png)
![Variance](/results/experiment2_MNIST/graphs/experiment2_variance_quantification.png)
![Individual Runs](/results/experiment2_MNIST/graphs/experiment2_individual_runs.png)

### Results

| Activation  | Mean Acc | Std   | Range | Mean Loss | Loss Std |
|-------------|----------|-------|--------|-----------|----------|
| ReLU        | 96.73 | 0.043 | 0.10 | 0.106 | 0.0007 |
| Sigmoid     | 92.50 | 0.123 | 0.30 | 0.284 | 0.0027 |
| Cubic-Node  | 87.62 | 0.156 | 0.34 | 0.404 | 0.0070 |
| Quad-Node   | 87.34 | 0.565 | 1.37 | 0.411 | 0.0183 |
| Cubic-Layer | 75.77 | 1.603 | 3.41 | 0.743 | 0.0337 |
| Quad-Layer  | 46.90 | 21.209 | 50.82 | 1.573 | 0.5999 |

---

### Key Findings

### 1. ReLU still outperforms polynomial activations  
Polynomial activations lag:
- 9% behind ReLU
- 4% behind Sigmoid

### 2. Shared coefficients fail completely  
- Quad Layer shows 40% accuracy drop  
- Std increases by 37x
- Accuracy range is 50% across runs  

Likely due to gradient conflict when all neurons share the same parameters.

### 3. Per-neuron coefficients are stable and effective  
- Comparable stability to Sigmoid  
- Significantly better than shared parameters  
- Accuracy was 87%

### 4. Quadratic vs Cubic  
Only a 0.28% difference and cubic is slightly more stable. But quadratic is good enough.

---

### MNIST Conclusions

- Polynomial activations can learn MNIST, but are not as good as ReLU.  
- Per-neuron coefficients are essential whilst shared parameters are unusable.  
- Quadratic polynomials are enough and higher degrees bring diminishing returns.

---

# Experiment 3: Synthetic Datasets

Tested on four clean mathematical regression tasks:

- Linear  
- Polynomial  
- Discontinuous  
- Oscillatory  

### Plots

![Heatmap of Final Test Loss](/results/experiment3_SyntheticData/graphs/experiment3_heatmap.png)

### Results

| Task         | Cubic-Node | ReLU   | Improvement |
|--------------|------------|--------|-------------|
| Linear       | 0.0018     | 0.0178 | 9.9× better |
| Polynomial   | 0.0171     | 0.0605 | 3.5× better |
| Discontinuous| 0.1018     | 0.1093 | 1.07× better|
| Oscillatory  | 0.0156     | 0.1426 | 9.1× better |

### Insights

- Polynomial activations work best at approximating functions.  
- They significantly outperform ReLU on tasks with clear mathematical structure.  
- Surprisingly, cubic polynomials achieve 10× lower loss on linear regression as they learn to behave linearly by adjusting coefficients.

---

# Cross-Experiment Summary

| Dataset Type | ReLU Performance | Polynomial Performance | Winner |
|--------------|------------------|------------------------|--------|
| MNIST (noisy, real-world) | Excellent | Weaker | ReLU |
| Synthetic (clean, mathematical) | Good | Much better | Polynomials |

### Interpretation  
- MNIST relies on local feature extraction, which ReLU handles naturally.  
- Synthetic tasks rely on smooth function approximation, which polynomials excel at.

---

# Overall Conclusions

### Strengths of Polynomial Activations
- Excellent for structured mathematical tasks  
- High expressiveness due to learnable parameters  
- Outperform ReLU on regression tasks  
- Per-neuron coefficients yield stable training  

### Weaknesses
- Underperform on real-world data like MNIST  
- Require LayerNorm to prevent divergence  
- Shared parameters are unstable and unusable  
- More expensive than simple activations like ReLU

---

## Outcomes

Polynomial activations are powerful for clean, structured, mathematical tasks but fall short on noisy, real-world datasets.  
They are an interesting direction for research into learnable, adaptive activation functions. Especially when combined with proper normalisation and per-neuron parameterisation.