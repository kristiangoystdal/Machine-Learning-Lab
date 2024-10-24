import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt# Calculate class weights (optional)
from sklearn.utils.class_weight import compute_class_weight

def create_crater_segmentation_model(input_shape=(48, 48, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder (Downsampling Path)
    # Block 1
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    # Block 2
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    # Block 3
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    # Bottleneck (Middle)
    bottleneck = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    bottleneck = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(bottleneck)

    # Decoder (Upsampling Path)
    # Block 3
    up3 = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(bottleneck)
    concat3 = layers.concatenate([up3, conv3])
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat3)
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    # Block 2
    up2 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(conv4)
    concat2 = layers.concatenate([up2, conv2])
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat2)
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    # Block 1
    up1 = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(conv5)
    concat1 = layers.concatenate([up1, conv1])
    conv6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat1)
    conv6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)

    # Output Layer
    output = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv6)

    model = models.Model(inputs, output)

    return model


# Create the model
model = create_crater_segmentation_model()
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

history = model.fit(X_train, Y_train, epochs=25, batch_size=32, validation_data=(X_test,Y_test))

test_predictions = model.predict(X_test)

test_predictions_binary = np.where(test_predictions > 0.3, 1, 0)  # Thresholding predictions to 0 or 1

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

print(f"True Positive: {100*true_positive/(true_positive+false_negative):.2f}%, True Negative {100*true_negative/(true_negative+false_positive):.2f}%")