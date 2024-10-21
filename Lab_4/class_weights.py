import numpy as np


def calculate_class_weights(y):
    # Calculate the class weights
    class_weights = {
        0: len(y) / np.sum(y == 0),
        1: len(y) / np.sum(y == 1),
    }

    print(f"Class weights: {class_weights}")

    return class_weights


import numpy as np


def calculate_class_weights_2(Y_train):
    # Flatten the target masks to count the frequency of each class (0 and 1)
    flat_y = Y_train.flatten()

    # Count occurrences of each class (0 = background, 1 = object/crater)
    class_counts = np.bincount(flat_y.astype(int))

    # Calculate weights inversely proportional to the frequency
    total_pixels = len(flat_y)
    class_weights = {
        i: total_pixels / (2 * class_counts[i]) for i in range(len(class_counts))
    }

    return class_weights
