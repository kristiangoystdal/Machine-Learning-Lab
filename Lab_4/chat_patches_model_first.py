from data_loader import load_data
from tensorflow.keras import models, layers  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np

# Load the data
Xtrain2_a, Xtrain2_b, Ytrain2_a, Ytrain2_b, Xtest2_a, Xtest2_b = load_data()

# Reshape Xtrain2_a to be 7x7 patches, and add a channel dimension for grayscale
Xtrain2_a_reshaped = Xtrain2_a.reshape(-1, 7, 7, 1)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    Xtrain2_a_reshaped,
    Ytrain2_a,
    test_size=0.2,
    random_state=42,
    stratify=Ytrain2_a,
)

# Build a simple CNN model
cnn_model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(7, 7, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),  # Binary classification
    ]
)

# Compile the model
cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model using the training data
history = cnn_model.fit(
    X_train, Y_train, epochs=10, batch_size=256, validation_data=(X_val, Y_val)
)

# Predict on the validation set
Y_val_pred = cnn_model.predict(X_val)

# Convert predictions to binary (0 or 1)
Y_val_pred_binary = np.where(Y_val_pred > 0.5, 1, 0)

# Calculate the balanced accuracy score
balanced_acc = balanced_accuracy_score(Y_val, Y_val_pred_binary)
print(f"Balanced Accuracy Score: {balanced_acc}")
