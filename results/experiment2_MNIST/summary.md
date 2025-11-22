# Experiment 2: Baseline Comparison on MNIST

## Results

| Activation  | Test Accuracy Mean | Test Accuracy Std | Test Accuracy Range | Test Loss Mean | Test Loss Std |
|-------------|--------------------|-------------------|---------------------|----------------|---------------|
| ReLU        | 96.73              | 0.043             | 0.10                | 0.106          | 0.000719      |        
| Sigmoid     | 92.50              | 0.123             | 0.30                | 0.284          | 0.002714      |        
| Cubic-Node  | 87.62              | 0.156             | 0.34                | 0.404          | 0.007024      |        
| Quad-Node   | 87.34              | 0.565             | 1.37                | 0.411          | 0.018343      |       
| Cubic-Layer | 75.77              | 1.603             | 3.41                | 0.743          | 0.033731      |        
| Quad-Layer  | 46.90              | 21.209            | 50.82               | 1.573          | 0.599937      |       

## Key Findings

### 1. Performance Gap
-   ReLU performs the best (96.73% vs 87.62% for best polynomial)
-   9.1% accuracy gap shows that polynomial activations aren't optimal and need some changes to improve performance
-   Cubic-Node has a range and standard deviation very close to sigmoid and ReLU showing it is very stable

### 2. Per Neuron vs Shared Parameters

The most significant result is that shared parameters are flawed for polynomial activation functions.

Performance degradation:
-   Quadratic: 87.34% (node) -> 46.90% (Layer) = 40.4% drop
-   Cubic: 87.62% (node) -> 75.77% (Layer) = 11.8% drop

Stability degradation:
-   Quadratic: 0.57% std (node) -> 21.21% std (Layer) = 37x increase
-   Cubic: 0.16% std (node) -> 1.60% std (Layer) = 10x increase

### 3. Enormous Variance in Quad-Layer

Quad-Layer has a 50.82% accuracy range across runs:
-   Best run: 72% accuracy
-   Worst run: 21% accuracy

Variance comparison:
-   Quad-Layer is 491x more variable then ReLU (21.21% vs 0.04%)
-   Cubic-Layer is 13x more variable than the best Node polynomial

## Interpretation

### Why Shared Parameters Fail

My hypothesis is that the gradients conflicted:
-   All neurones in a layer share the same polynomial parameters
-   Different neurones would need different activation shapes
-   These conflicting needs would lead to conflicting gradient signals
-   This leads to unstable updates

### Why Node configurations Succeed

Per neuron parameters allow:
-   Each neuron to adapt its own activation and outputs
-   Independent optimisation (no gradient conflicts)
-   Increased capacity to learn due to extra parameters
-   Stable training

### Degree Comparison (Node only)

Interestingly, cubic and quadratic show similar performance:
-   Cubic-Node: 87.62% +- 0.16%
-   Quad-Node: 87.34% +- 0.57%
-   Difference: Only 0.28%

This shows that:
-   Higher degree polynomials provide minimal benefit
-   Quadratic may be enough for most tasks
-   Cubic has slightly better stability (0.16% vs 0.57% std)

### Comparison to Standard Activations

Performance ranking:
1.  ReLU: 96.73% (baseline)
2.  Sigmoid: 92.50% (-4.2% from ReLU)
3.  Polynomial Node: 87.34% - 87.62% (-9.2% from ReLU)

Stability ranking:
1.  ReLU: 0.04% std
2.  Sigmoid: 0.12% std
3.  Cubic-Node: 0.16% std
4.  Quad-Node: 0.57% std

Polynomial activations with per neuron parameters achieve stability comparable to standard activations, despite being learnable and more complex.

## Conclusions

### Main Findings
1.  Polynomial activations are viable: achieving 87% on MNIST with stable training
2.  Shared parameters are unsuitable: 40% performance drop and 37x variance
3.  Per neurone parameters: boosts performance and stability
4.  Degree 2 sufficient: cubic offers small improvement over quadratic
5.  Performance gap: 9% behind ReLU suggests there is room for improvement

### What I found surprising
1.  ReLU being the simplest activation function outperforms both sigmoid and polynomial activations even though they are much more complex.
2.  Cubic-Node training is more stable then Quad-Node, which I didn't expect since a cubic equation grows faster than a quadratic. Although LayerNorm might have prevented this issue.
3.  Shared parameters performed worse than per node. Which is surprising because I thought that because each layer updates the coefficients (a,b,c) using the average gradient of the whole layer they would update smoothly allowing the model to more efficiently learn.