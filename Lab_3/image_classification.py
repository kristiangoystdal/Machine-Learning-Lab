import numpy as np
import matplotlib.pyplot as plt

# Load the flattened data
X = np.load('Xtrain1.npy')  # Assuming the file is named Xtrain1.npy

# Reshape the data into (n_samples, 48, 48, 1)
X = X.reshape(-1, 48, 48, 1)

# Normalize pixel values (assuming they are in range [0, 255])
X = X / 255.0

# print(X.shape)

# random_indices = np.random.choice(X.shape[0], 10, replace=False)

# # Set up the plot
# plt.figure(figsize=(10, 10))

# for i, idx in enumerate(random_indices):
#     plt.subplot(2, 5, i + 1)  # Create a grid of 2 rows and 5 columns
#     plt.imshow(X[idx].reshape(48, 48), cmap='gray')  # Reshape to (48, 48) for visualization
#     plt.axis('off')  # Turn off axis labels
#     plt.title(f'Image {idx}')  # Optional: title with the index of the image

# plt.tight_layout()
# plt.show()



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

input_shape = (48, 48, 1)
output_shape = (1,)  # For binary output

# Create the CNN model
model = Sequential()

# Convolutional layer 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization

# Output layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Load the labels
y = np.load('ytrain1.npy')

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Evaluate on validation data
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f'Validation accuracy: {val_acc}')

# Predict on new data (X_test needs to be reshaped similarly)
X_test = np.load('Xtrain1_extra.npy')
X_test = X_test.reshape(-1, 48, 48, 1) / 255.0
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype(int)

predicted_classes = predicted_classes.reshape(835,)

print('Predictions shape: ', predicted_classes.shape)

Y_test = np.load('ytrain1_extra.npy')

print('Fasit shape: ', Y_test.shape)

Successes = 0

Failures = 0

for i in range(predicted_classes.size):
    if predicted_classes[i] == Y_test[i]:
        Successes += 1
    else:
        Failures += 1

print('Successes: ',Successes)
print('Failures: ',Failures)