import numpy as np
import time
from src.NNClass import NN

def createSyntheticData(task, num_samples=10000):
    inputs = np.random.uniform(-1, 1, size=(num_samples, 1))
    if task == "linear":
        labels = 2 * inputs + 1
    elif task == "polynomial":
        labels = inputs**3 - 2*inputs**2 + inputs
    elif task == "discontinuous":
        labels = np.where(inputs > 0, 1, -1)
    elif task == "oscillatory":
        labels = np.sin(5 * inputs)

    splitIndex = int(num_samples * 0.8)

    trainInputs = inputs[:splitIndex]
    trainLabels = labels[:splitIndex]
    testInputs = inputs[splitIndex:]
    testLabels = labels[splitIndex:]

    return trainInputs, trainLabels, testInputs, testLabels

tasks = ["linear", "polynomial", "discontinuous", "oscillatory"]

numRuns = 3
epochs = 50
learningRate = 1e-3
batchSize = 32
layerSizes = [1, 16, 1]  

activations = [
    {"name": "ReLU",        "activation": "relu",  "polynomialType": None},
    {"name": "Sigmoid",     "activation": "sigmoid","polynomialType": None},
    {"name": "Quad-Node",   "activation": "quad",  "polynomialType": "node"},
    {"name": "Cubic-Node",  "activation": "cubic", "polynomialType": "node"},
]

results = {}
seeds = [49, 124, 64]

for task in tasks:
    np.random.seed(5600) # for the synthetic data
    trainInputs, trainLabels, testInputs, testLabels = createSyntheticData(task)
    
    results[task] = {}
    
    for act in activations:
        activation = act["activation"]
        polynomialType = act["polynomialType"]
        activationName = act["name"]
        
        results[task][activationName] = {        
            "train_loss": [],
            "test_loss": [],
            "learned_params": None,
        }

        for run in range(numRuns):
            seed = seeds[run]
            np.random.seed(seed)

            model = NN(
                polynomialType=polynomialType,
                layerSizes=layerSizes,
                batchSize=batchSize,
                activationFunction=activation,
                normalisation="layer",
                outputActivation="linear",
                lossFunction="mse",
            )
            model.createWeightsAndBiases()

            run_train_loss = []
            run_test_loss = []

            epochStart = time.time()

            for epoch in range(1, epochs + 1):
                _, train_loss = model.train(trainInputs, trainLabels, learningRate)
                
                test_outputs = model.forward(testInputs)
                test_loss = model.MSE(test_outputs, testLabels)
                
                run_train_loss.append(train_loss)
                run_test_loss.append(test_loss)

            epochEnd = time.time()
            print(f"Run: {run+1}, Took: {(epochEnd - epochStart):.2f}s, Train loss: {run_train_loss[-1]:.4f}, Test loss: {run_test_loss[-1]:.4f}")

            results[task][activationName]["train_loss"].append(run_train_loss)
            results[task][activationName]["test_loss"].append(run_test_loss)

            if(run == 0):
                learned_params = {
                    "weights": [w.copy() for w in model.weights],
                    "biases": [b.copy() for b in model.biases],
                }
                if(activation == "quad" or activation == "cubic"):
                    learned_params["a"] = [a.copy() for a in model.a]
                    learned_params["b"] = [b.copy() for b in model.b]
                    learned_params["c"] = [c.copy() for c in model.c]
                    if(activation == "cubic"):
                        learned_params["d"] = [d.copy() for d in model.d]
                results[task][activationName]["learned_params"] = learned_params

np.savez("results/experiment3_SyntheticData/experiment3_SyntheticData.npz", **results)
