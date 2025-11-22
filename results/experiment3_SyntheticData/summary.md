# Experiment 3: Synthetic Datatset

## Key Findings

### 1. Superiority of Polynomial Activations on Synthetic Tasks

Cubic-Node acheived the best performance on all four synthetic tasks, showing that learned polynomial activations can outperform fixed activations when the task has a mathematical structure.

| Task         | Cubic-Node | ReLU   | Improvement  |
|--------------|------------|--------|--------------|
| Linear       | 0.0018     | 0.0178 | 9.9x better  |
| Polynomial   | 0.0171     | 0.0605 | 3.5x better  |
| Discontinous | 0.1018     | 0.1093 | 1.07x better |
| Oscillatory  | 0.0156     | 0.1426 | 9.1x better  |

Which I find surprising since the universal approximation theory says that a 2 layer FFN can approximate most continous functions only if the activation function is not a polynomial. Although this might be due to overfitting as the universal approxmitation theory doesnt say a polynomial activation function cannot approximate a function if it overfits the dataset.

### 2. Polynomials Excel on Linear Tasks

Another thing I found quite surprising was that Cubic-Node acheived 9.9x lower loss than ReLU on linear regression (0.0018 vs 0.0178). This shows that polynomial activations can learn to behave linearly. This capability to adapt provides and advantage over ReLU's fixed slope.

## Comparison to MNIST (experiment 2)

| Dataset                         | ReLU   | Polynomial | Gap    |
|---------------------------------|--------|------------|--------|
| MNIST (complex, real world)     | 96.7%  | 87.6%      | -9.1%  |
| Synthetic (clean, mathematical) | Varies | Varies     | 7-160% |

From this we can see that polynomial activations work best when tasks have clear mathematical structure and the data has no noise. Which might be the reason for the gap on MNIST as it's complex and isnt based on trying to approximate a function but learning local feautures.


    