import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Load the augmented training data
X_train_smote = np.load("X_train_augmented.npy")
y_train_smote = np.load("y_train_augmented.npy")

# Convert labels to categorical (0 or 1)
y_train_smote_cat = to_categorical(y_train_smote)

# Split the augmented dataset into training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_smote, y_train_smote_cat, test_size=0.2, random_state=42
)

# Build the CNN model
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),  # Output layer for binary classification
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model using the augmented data
model.fit(
    X_train_split,
    y_train_split,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
)

# Make predictions on the validation set
y_val_pred_probs = model.predict(X_val)

# Convert predicted probabilities to class labels
y_val_pred = np.argmax(y_val_pred_probs, axis=1)
y_val_true = np.argmax(
    y_val, axis=1
)  # Convert validation set labels from one-hot to class labels

# Calculate the F1 score
f1 = f1_score(y_val_true, y_val_pred)

print(f"F1 Score on the validation set: {f1}")

# Find misclassified images
misclassified_idx = np.where(y_val_true != y_val_pred)[0]

# Plot misclassified images
if len(misclassified_idx) > 0:
    print(f"Number of misclassified images: {len(misclassified_idx)} of {len(y_val)}")
    fig, axes = plt.subplots(
        3, 7, figsize=(15, 12)
    )  # 3 rows, 7 columns to display 21 images

    for i, idx in enumerate(
        misclassified_idx[:21]
    ):  # Show up to 21 misclassified images
        ax = axes[i // 7, i % 7]
        ax.imshow(X_val[idx].reshape(48, 48), cmap="gray")
        ax.set_title(f"True: {y_val_true[idx]}, Pred: {y_val_pred[idx]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("No misclassified images.")


# Save the model
# model.save("cnn_model_smote.h5")
# print("CNN model trained with data and saved as 'cnn_model_smote.h5'.")

from tensorflow.keras.utils import plot_model

# Plot the model architecture
plot_model(model, to_file="cnn_model.png", show_shapes=True, show_layer_names=True)

# Optionally, display the model image directly in a notebook environment (like Jupyter or Colab)
from IPython.display import Image

Image(filename="cnn_model.png")
