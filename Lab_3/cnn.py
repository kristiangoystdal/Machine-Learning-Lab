import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from keras.callbacks import Callback
import ipywidgets as widgets
from IPython.display import display
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from models import build_resnet_model


def load_data():
    x_train_path = "Xtrain1.npy"
    y_train_path = "ytrain1.npy"
    x_test_path = "Xtest1.npy"

    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    x_test = np.load(x_test_path)

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)

    return x_train, y_train, x_test


def build_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(2, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def calculate_final_f1_score(model, x_val, y_val):
    # Make predictions on the validation set
    y_val_pred = np.argmax(model.predict(x_val), axis=1)

    # Calculate F1 score
    f1 = f1_score(y_val, y_val_pred, average="weighted")
    print(f"Final F1 Score on Validation Set: {f1:.4f}")


def train_model(model, x_train, y_train):
    x_train = x_train.reshape(-1, 48, 48, 1)
    y_train = y_train.reshape(-1, 1)

    y_train = to_categorical(y_train)

    # Split the validation set manually (20% split)
    split_idx = int(0.8 * x_train.shape[0])
    x_val, y_val = x_train[split_idx:], y_train[split_idx:]
    x_train, y_train = x_train[:split_idx], y_train[:split_idx]

    # Train the model with validation data passed to both the model and the callback
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        class_weight={0: 1.3835227272727273, 1: 0.7829581993569131},
    )

    # Calculate the final F1 score after training
    # calculate_final_f1_score(model, x_val, y_val)


def plot_image_slider(x_test, y_pred_labels):
    def view_image(index):
        plt.figure(figsize=(5, 5))
        image = x_test[index].reshape(48, 48)
        plt.imshow(image, cmap="gray")
        plt.title(f"Pred: {y_pred_labels[index]}")
        plt.axis("off")
        plt.show()

    slider = widgets.IntSlider(min=0, max=len(x_test) - 1, step=1, description="Index")
    widgets.interact(view_image, index=slider)


def main():
    x_train, y_train, x_test = load_data()

    model = build_model()
    # model = build_resnet_model()
    train_model(model, x_train, y_train)

    # Predict on the test data
    x_test = x_test.reshape(-1, 48, 48, 1)
    # y_pred = model.predict(x_test)
    # y_pred_labels = np.argmax(y_pred, axis=1)

    # Display the images with a slider
    # plot_image_slider(x_test, y_pred_labels)


if __name__ == "__main__":
    main()
