import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt# Calculate class weights (optional)
from sklearn.utils.class_weight import compute_class_weight






# Define the CNN architecture for image segmentation
def create_segmentation_model(input_shape=(48, 48, 1)):
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))

    # First Convolutional Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolutional Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third Convolutional Layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Upsampling and Convolutional Layers for Reconstruction (Deconvolution part)
    model.add(layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))

    model.add(layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu'))

    # Output Layer for Segmentation (1 channel, 0 or 1)
    model.add(layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'))

    return model

# Create the model
model = create_segmentation_model()
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load your data
X_train = np.load('Xtrain2_b.npy')  # Replace with actual dataset
Y_train = np.load('Ytrain2_b.npy')  # Replace with actual dataset

# Check the shapes
print("X_train shape:", X_train.shape)  # Ensure it shows (num_samples, 48, 48)
print("Y_train shape:", Y_train.shape)  # Ensure it shows (num_samples, 48, 48)

# Reshape X_train and Y_train if necessary
if X_train.ndim == 2:  # If the input shape is (num_samples, 2304)
    X_train = X_train.reshape((-1, 48, 48, 1))
if Y_train.ndim == 2:  # If the output shape is (num_samples, 2304)
    Y_train = Y_train.reshape((-1, 48, 48, 1))

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

model = create_segmentation_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=25, batch_size=32, validation_split=0.2)

test_predictions = model.predict(X_test)
test_predictions_binary = np.round(test_predictions)  # Thresholding predictions to 0 or 1

# Flatten the arrays for calculating balanced accuracy
Y_test_flat = Y_test.flatten()
test_predictions_flat = test_predictions_binary.flatten()

# Calculate balanced accuracy
balanced_acc = balanced_accuracy_score(Y_test_flat, test_predictions_flat)

print(f"Balanced Accuracy: {balanced_acc}")

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for i in range(Y_test_flat.size):
    if Y_test_flat[i] == 1 and test_predictions_flat[i] == 1:
        true_positive += 1
    elif Y_test_flat[i] == 0 and test_predictions_flat[i] == 0:
        true_negative += 1
    elif Y_test_flat[i] == 0 and test_predictions_flat[i] == 1:
        false_positive += 1
    else:
        false_negative += 1

print(f"True Positive: {true_positive}, False Positive: {false_positive}, True Negative {true_negative}, False Negative {false_negative}")