import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load results
results = np.load("results/experiment2_MNIST/experiment2_MNIST.npz", allow_pickle=True)
results = {k: v.item() for k, v in results.items()}  # convert back to dict

saveDir = "results/experiment2_MNIST/graphs/experiment2"

colors = ["r", "g", "b", "c", "m", "y"]

def errorbarPlot(data, title, ylabel, filename):
    plt.figure(figsize=(12, 5))
    for i, normName in enumerate(data.keys()):
        runs = np.array(data[normName])
        mean = np.mean(runs, axis=0)
        std = np.std(runs, axis=0)
        plt.errorbar(range(len(mean)), mean, yerr=std, label=normName, color=colors[i], fmt='-o', markersize=3, capsize=3, elinewidth=1)
        
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{saveDir}_{filename}", dpi=300)
    plt.close()

errorbarPlot({k: results[k]["train_accuracy"] for k in results.keys()}, "Train Accuracy per Epoch", "Accuracy (%)", "train_accuracy.png")
errorbarPlot({k: results[k]["train_loss"] for k in results.keys()}, "Train Loss per Epoch", "Loss", "train_loss.png")
errorbarPlot({k: results[k]["test_accuracy"] for k in results.keys()}, "Test Accuracy per Epoch", "Accuracy (%)", "test_accuracy.png")
errorbarPlot({k: results[k]["test_loss"] for k in results.keys()}, "Test Loss per Epoch", "Loss", "test_loss.png")

summary = []
for normName in results.keys():
    final_train_acc = np.mean([run[-1] for run in results[normName]["train_accuracy"]])
    final_test_acc = np.mean([run[-1] for run in results[normName]["test_accuracy"]])
    final_train_loss = np.mean([run[-1] for run in results[normName]["train_loss"]])
    final_test_loss = np.mean([run[-1] for run in results[normName]["test_loss"]])
    
    summary.append({
        "Normalisation": normName,
        "Final Train Acc (%)": final_train_acc,
        "Final Test Acc (%)": final_test_acc,
        "Final Train Loss": final_train_loss,
        "Final Test Loss": final_test_loss,
    })

summary = pd.DataFrame(summary)
summary.to_csv(f"{saveDir}_summaryTable.csv", index=False)

print(summary)