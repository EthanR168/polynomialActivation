import numpy as np
import time
from src.NNClass import NN

trainImages = np.load("MNIST/train_images.npy")  # shape (60000, 784)
trainLabels = np.load("MNIST/train_labels.npy")  # shape (60000, 10)
testImages = np.load("MNIST/test_images.npy")  # shape (10000, 784)
testLabels = np.load("MNIST/test_labels.npy")  # shape (10000, 10)

numRuns = 3
epochs = 50
learningRate = 1e-3
batchSize = 32
layerSizes = [784, 128, 64]

activations = [
    {"name": "Quad-Node",   "activation": "quad",    "polynomialType": "node"},
    {"name": "Quad-Layer",  "activation": "quad",    "polynomialType": "layer"},
    {"name": "Cubic-Node",  "activation": "cubic",   "polynomialType": "node"},
    {"name": "Cubic-Layer", "activation": "cubic",   "polynomialType": "layer"},
    {"name": "ReLU",        "activation": "relu",    "polynomialType": None},
    {"name": "Sigmoid",     "activation": "sigmoid", "polynomialType": None},
]

results = {}
seeds = [49, 124, 64, 1023, 964]

for act in activations:
    activation = act["activation"]
    polynomialType = act["polynomialType"]
    activationName = act["name"]
    
    results[activationName] = {        
        "train_accuracy": [],
        "train_loss": [],
        "test_accuracy": [],
        "test_loss": [],
    }

    for run in range(numRuns):
        seed = seeds[run]
        
        np.random.seed(seed)

        model = NN(
            polynomialType=polynomialType,
            layerSizes=layerSizes,
            batchSize=batchSize,
            activationFunction=activation,
            normalisation="layer"
        )
        model.createWeightsAndBiases()

        run_train_accuracy = []
        run_train_loss = []
        run_test_accuracy = []
        run_test_loss = []

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

        epochEnd = time.time()
        
        print(f"Run: {run+1}, took: {(epochEnd - epochStart):.2f}s, train acc: {run_train_accuracy[-1]}%, test acc: {run_test_accuracy[-1]}%")
            
        results[activationName]["train_accuracy"].append(run_train_accuracy)
        results[activationName]["train_loss"].append(run_train_loss)
        results[activationName]["test_accuracy"].append(run_test_accuracy)
        results[activationName]["test_loss"].append(run_test_loss)

# Save results 
np.savez("results/experiment2_MNIST/experiment2_MNIST.npz", **results)
