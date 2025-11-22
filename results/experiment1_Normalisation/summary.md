# Experiment 1: Different Normalisation Strategies for Polynomial Activations

## Results

| Normalisation | Train Accuracy | Test Accuracy | Train Loss         | Test Loss          | Converged |
|---------------|----------------|---------------|--------------------|--------------------|-----------|
| BatchNorm     | 81.75          | 84.74         | 0.5837565890483374 | 0.4987542558419846 | True      | 
| LayerNorm     | 83.77          | 83.84         | 0.5271809028952171 | 0.5171281860847126 | True      | 
| tanh          | 74.05          | 74.42         | 0.8147049566539063 | 0.7980804478199707 | True      |          
| clip          | 70.41          | 70.46         | 0.9556437199692019 | 0.9494655719409928 | True      |            
| No Norm       | 9.87           | 9.80          | NaN                | NaN                | False     |            

## Key Findings

1.  **Normalisation is needed:** Without it, polynomial activations fail due to explosive growth across layers. Which led to overflow errors when running no normalisation.
2.  **Simple bounding helps:** Tanh and clipping solves the problems of extreme outputs and allows training to work (74%), but they don't stabilise outputs across layers.
3.  **Full normalisation is best:** BatchNorm and LayerNorm significantly improves stability and accuracy (82-84)%.

## Conclusion

**Layer Normalisation** is the best normalisation for polynomial activation functions as it provides:
-   Complete stability (no overflow)
-   Best accuracy (84%)
-   Independence from batch size (unlike BatchNorm)

## Why Not Combine Strategies?

I considered combining approaches (e.g., TanH + LayerNorm) but realised it would be redundant because:
-   **Tanh + LayerNorm:** TanH bounds the inputs between -1 and 1, but LayerNorm normalises the inputs anyway, making the bounding redundant.
-   **Clip + Layer Norm:** Clipping constrains variance, which interferes with LayerNorm's statistical normalisation which depends on the variance.

Overall the strategies work through different means and combining them wouldn't have any benefit apart from longer runtime due to extra operations and derivatives needed to be calculated.
