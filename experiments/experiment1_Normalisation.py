import numpy as np
import time
from src.NNClass import NN

trainImages = np.load("MNIST/train_images.npy")  # shape (60000, 784)
trainLabels = np.load("MNIST/train_labels.npy")  # shape (60000, 10)
testImages = np.load("MNIST/test_images.npy")  # shape (10000, 784)
testLabels = np.load("MNIST/test_labels.npy")  # shape (10000, 10)

numRuns = 3
epochs = 20
learningRate = 1e-3
batchSize = 32
layerSizes = [784, 128, 64, 10]

normalisationTypes = [
    {"name": "batchNorm", "type": "batch"},
    {"name": "layerNorm", "type": "layer"},
    {"name": "tanh",      "type": "tanh"},
    {"name": "clip",      "type": "clip"},
    {"name": "No Norm",   "type": "none"},
]

results = {}
seeds = [49, 124, 64]

for norm in normalisationTypes:
    normType = norm["type"]
    normName = norm["name"]
    
    results[normName] = {        
        "train_accuracy": [],
        "train_loss": [],
        "test_accuracy": [],
        "test_loss": [],
        "converged": [], # whilst training did it get a NaN (Not A Number) 
    }

    for run in range(numRuns):
        seed = seeds[run]
        
        np.random.seed(seed)

        model = NN(
            polynomialType="node",
            layerSizes=layerSizes,
            batchSize=batchSize,
            activationFunction="quad",
            normalisation=normType,
        )
        model.createWeightsAndBiases()

        run_train_accuracy = []
        run_train_loss = []
        run_test_accuracy = []
        run_test_loss = []
        run_converged = []

        epochStart = time.time()

        for epoch in range(1, epochs + 1):
            train_accuracy, train_loss = model.train(trainImages, trainLabels, learningRate)
            
            test_outputs = model.forward(testImages)
            test_loss = model.CrossEntropyLossFunction(test_outputs, testLabels)
            test_predictions = np.argmax(test_outputs, axis=1)
            test_true = np.argmax(testLabels, axis=1)
            test_acc = (np.sum(test_predictions == test_true) / len(testLabels)) * 100
            
            run_train_accuracy.append(train_accuracy)
            run_train_loss.append(train_loss)
            run_test_accuracy.append(test_acc)
            run_test_loss.append(test_loss)

            if(np.isnan(train_loss) or np.isnan(test_loss)):
                run_converged.append(False)
            else:
                run_converged.append(True)

        epochEnd = time.time()
        
        print(f"Run: {run+1}, took: {(epochEnd - epochStart):.2f}s, train acc: {run_train_accuracy[-1]}%, test acc: {run_test_accuracy[-1]}%, converged: {run_converged[-1]}")
            
        results[normName]["train_accuracy"].append(run_train_accuracy)
        results[normName]["train_loss"].append(run_train_loss)
        results[normName]["test_accuracy"].append(run_test_accuracy)
        results[normName]["test_loss"].append(run_test_loss)
        results[normName]["converged"].append(run_converged)

# Save results 
np.savez("results/experiment1_Normalisation/experiment1_Normalisation.npz", **results)
