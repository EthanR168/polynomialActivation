import numpy as np
from torchvision import datasets

# Load MNIST data
trainset = datasets.MNIST(root='MNIST/', train=True, download=True)
testset = datasets.MNIST(root='MNIST/', train=False, download=True)

# Preprocess and save the data as arrays
train_images = trainset.data.view(-1, 28*28).numpy().astype('float16') / 255.0  # Normalize pixel values to [0,1]
train_labels = np.eye(10, dtype='float16')[trainset.targets.numpy()]          # One-hot encode labels

test_images = testset.data.view(-1, 28*28).numpy().astype('float16') / 255.0
test_labels = np.eye(10, dtype='float16')[testset.targets.numpy()]

# save to disk for reuse
np.save('MNIST/train_images.npy', train_images)
np.save('MNIST/train_labels.npy', train_labels)
np.save('MNIST/test_images.npy', test_images)
np.save('MNIST/test_labels.npy', test_labels)