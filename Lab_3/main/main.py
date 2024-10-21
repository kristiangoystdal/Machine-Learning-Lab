import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time


from model import build_model
from data_augmentation import augment_data, flatten_images
from plot_predictions import plot_images, plot_list, plot_average


# Load the training data
X_train = np.load("Xtrain1.npy")
y_train = np.load("Ytrain1.npy")

X_train_extra = np.load("Xtrain1_extra.npy")

X_test = np.load("Xtest1.npy")

X_train_augmented, Y_train_augmented = augment_data(X_train, y_train)

# Print the shapes of the augmented data
print(X_train_augmented.shape, Y_train_augmented.shape)
print(X_train_extra.shape)

# Normalize the pixel values
X_train_augmented = X_train_augmented / 255.0
X_train_extra = X_train_extra / 255.0

# Convert the labels to one-hot encoding
Y_train_augmented = to_categorical(Y_train_augmented, num_classes=2)

# Build the CNN model
model = build_model()

x_train_loop = X_train_augmented
y_train_loop = Y_train_augmented

x_train_extra_loop = X_train_extra

f1_scores = []
loss = []
accuracy = []
val_loss = []
val_accuracy = []

n_predictions = 5
i = 0
iteration_times = []  # List to keep track of iteration times
while True:

    start_time = time.time()
    n_iterations = x_train_extra_loop.shape[0] // n_predictions

    x_train_loop = x_train_loop.reshape(-1, 48, 48, 1)
    x_train_extra_loop = x_train_extra_loop.reshape(-1, 48, 48, 1)

    # Split the augmented data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        x_train_loop, y_train_loop, test_size=0.2, random_state=42
    )

    # Train the model
    history = model.fit(
        X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_val, Y_val)
    )

    loss.append(history.history["loss"])
    accuracy.append(history.history["accuracy"])
    val_loss.append(history.history["val_loss"])
    val_accuracy.append(history.history["val_accuracy"])

    # Predict the first n images in the extra dataset
    predictions = model.predict(x_train_extra_loop[:n_predictions])

    # Convert the predictions to class labels
    predictions = np.argmax(predictions, axis=1)

    # Convert the predictions to one-hot encoding
    predictions = to_categorical(predictions, num_classes=2)

    # Print the shapes
    print(x_train_loop.shape, y_train_loop.shape)
    print(x_train_extra_loop.shape, predictions.shape)

    # Combine the augmented and extra datasets
    x_train_loop = flatten_images(x_train_loop)
    x_train_extra_loop = flatten_images(x_train_extra_loop)
    x_train_loop = np.vstack([x_train_loop, x_train_extra_loop[:n_predictions]])

    # Add predictions to the augmented labels
    y_train_loop = np.concatenate([y_train_loop, predictions])

    # Remove the first n images from the extra dataset
    x_train_extra_loop = x_train_extra_loop[n_predictions:]

    # Print F1 score
    y_val_pred_probs = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
    y_val_true = np.argmax(Y_val, axis=1)
    f1 = f1_score(y_val_true, y_val_pred, average="binary")
    print(f"F1 Score on the validation set: {f1}")
    f1_scores.append(f1)

    if f1_scores[-1] > 0.97:
        n_predictions = 10

    # Check if there are any images left in the extra dataset
    if len(x_train_extra_loop) == 0:
        break

    # Calculate time for the current iteration
    end_time = time.time()
    iteration_time = end_time - start_time
    iteration_times.append(iteration_time)

    # Calculate the average time per iteration
    avg_iteration_time = np.mean(iteration_times)

    # Calculate time remaining based on the average time and iterations left
    iterations_remaining = n_iterations - 1
    time_remaining = avg_iteration_time * iterations_remaining
    minutes, seconds = divmod(time_remaining, 60)
    print(f"Time remaining: {int(minutes)} minutes and {int(seconds)} seconds")

    i += 1

plot_list(f1_scores)

X_test = X_test / 255.0

X_test = X_test.reshape(-1, 48, 48, 1)

# Predict the test set
y_test_pred_probs = model.predict(X_test)

# Convert predicted probabilities to class labels
y_test_pred = np.argmax(y_test_pred_probs, axis=1)

# Save the predictions to a file
np.save("Ytest1_pred.npy", y_test_pred)
