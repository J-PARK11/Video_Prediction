
import os
import numpy as np

def load_mmnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'moving_mnist/mnist_test_seq.npy')
    dataset = np.load(path)
    
     # Swap the axes representing the number of frames and number of data samples.
    dataset = np.swapaxes(dataset, 0, 1)

    # Add a channel dimension since the images are grayscale.
    dataset = np.expand_dims(dataset, axis=-1)

    print("(samples, frames, w, h, c) :",dataset.shape)
    return dataset