import numpy as np


def load_data():
    # Load the data
    Xtrain2_a = np.load("Xtrain2_a.npy")
    Xtrain2_b = np.load("Xtrain2_b.npy")
    Ytrain2_a = np.load("Ytrain2_a.npy")
    Ytrain2_b = np.load("Ytrain2_b.npy")

    Xtest2_a = np.load("Xtest2_a.npy")
    Xtest2_b = np.load("Xtest2_b.npy")

    return Xtrain2_a, Xtrain2_b, Ytrain2_a, Ytrain2_b, Xtest2_a, Xtest2_b
