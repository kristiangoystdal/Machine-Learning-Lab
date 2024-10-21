from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras import models, layers
import numpy as np

from data_loader import load_data
from class_weights import *

# Load the data
Xtrain2_a, Xtrain2_b, Ytrain2_a, Ytrain2_b, Xtest2_a, Xtest2_b = load_data()

# Reshape Xtrain2_b and Ytrain2_b back to their 48x48 pixel format
Xtrain2_b_reshaped = Xtrain2_b.reshape(-1, 48, 48, 1)  # Adding channel for grayscale
Ytrain2_b_reshaped = Ytrain2_b.reshape(-1, 48, 48, 1)  # Adding channel for mask labels

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    Xtrain2_b_reshaped, Ytrain2_b_reshaped, test_size=0.2, random_state=42
)

# Build a CNN model for segmentation
cnn_model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.UpSampling2D((2, 2)),  # Upsample back to 48x48
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(
            1, (1, 1), activation="sigmoid"
        ),  # Output 48x48 mask with binary values
    ]
)

# Compile the model
cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model using the training data
history = cnn_model.fit(
    X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_val, Y_val)
)


# Predict on the validation set
Y_val_pred = cnn_model.predict(X_val)

# Convert predictions to binary (0 or 1)
Y_val_pred_binary = np.where(Y_val_pred > 0.5, 1, 0)

# Calculate the balanced accuracy score
balanced_acc = balanced_accuracy_score(Y_val.flatten(), Y_val_pred_binary.flatten())
print(f"Balanced Accuracy Score: {balanced_acc}")
