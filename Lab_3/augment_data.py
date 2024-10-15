import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def apply_smote(X_train, y_train):
    # Flatten the images to 1D vectors for SMOTE
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    # Apply SMOTE to the flattened data
    smote = SMOTE(random_state=42)
    X_train_flat_smote, y_train_smote = smote.fit_resample(X_train_flat, y_train)

    # Reshape the oversampled data back to the original image shape
    X_train_smote = X_train_flat_smote.reshape(-1, 48, 48, 1)

    return X_train_smote, y_train_smote


def apply_image_augmentation(X_train, y_train):
    # Reshape data back to 4D (samples, height, width, channels)
    X_train_reshaped = X_train.reshape(
        -1, 48, 48, 1
    )  # Assuming 48x48 images with 1 channel

    # Initialize ImageDataGenerator with augmentations
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    # Fit the generator to the training data
    datagen.fit(X_train_reshaped)

    # Generate augmented images
    X_augmented, y_augmented = next(
        datagen.flow(X_train_reshaped, y_train, batch_size=len(X_train))
    )

    return X_augmented, y_augmented


def apply_gaussian_noise(X_train, y_train, mean=0, std=0.1):
    # Reshape data back to 4D (samples, height, width, channels)
    X_train_reshaped = X_train.reshape(
        -1, 48, 48, 1
    )  # Assuming 48x48 images with 1 channel

    # Apply Gaussian noise
    noise = np.random.normal(mean, std, X_train_reshaped.shape)
    X_augmented = X_train_reshaped + noise

    # Ensure pixel values are still between 0 and 1
    X_augmented = np.clip(X_augmented, 0, 1)

    return X_augmented, y_train


def choose_augmentation_method():
    print("Choose augmentation method:")
    print("1. SMOTE")
    print("2. ImageDataGenerator (Rotation, Shift, Zoom, Flip)")
    print("3. Gaussian Noise Augmentation")

    choice = input("Enter 1, 2, or 3: ")
    return choice


# Load the training data
X_train = np.load("Xtrain1.npy")
y_train = np.load("ytrain1.npy")

# Normalize the images
X_train = X_train / 255.0

# Ask the user to choose the augmentation method
choice = choose_augmentation_method()

if choice == "1":
    # Apply SMOTE
    X_train_augmented, y_train_augmented = apply_smote(X_train, y_train)
    print("SMOTE augmentation complete.")
elif choice == "2":
    # Apply Image Augmentation
    X_train_augmented, y_train_augmented = apply_image_augmentation(X_train, y_train)
    print("ImageDataGenerator augmentation complete.")
elif choice == "3":
    # Apply Gaussian Noise Augmentation
    X_train_augmented, y_train_augmented = apply_gaussian_noise(X_train, y_train)
    print("Gaussian noise augmentation complete.")
else:
    print("Invalid choice. Exiting.")
    exit()

# Save the augmented dataset
np.save("X_train_augmented.npy", X_train_augmented)
np.save("y_train_augmented.npy", y_train_augmented)

print("Shape of augmented data:", X_train_augmented.shape, y_train_augmented.shape)

print(
    "Augmentation complete. Data saved to 'X_train_augmented.npy' and 'y_train_augmented.npy'."
)

# Plot 10 of the augmented images with their labels
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(10):
    # Display the augmented images
    axes[i // 5, i % 5].imshow(X_train_augmented[i].reshape(48, 48), cmap="gray")
    axes[i // 5, i % 5].set_title(f"Label: {y_train_augmented[i]}")
    axes[i // 5, i % 5].axis("off")

plt.tight_layout()
plt.show()
