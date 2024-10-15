from sklearn.utils import resample
import cv2
import random
import numpy as np

# Load the datasets
image_file_path = 'Xtrain1.npy'
labels_file_path = 'Ytrain1.npy'

# Load the images (flattened 48x48 grayscale) and the class labels (0 or 1)
X_train = np.load(image_file_path)
Y_train = np.load(labels_file_path)

# Check shapes and inspect the dataset
X_train.shape, Y_train.shape, np.unique(Y_train, return_counts=True)  # Checking the class distribution as well

# Reshape flattened images back to 48x48
def reshape_images(X):
    return X.reshape(-1, 48, 48)

def flatten_images(X):
    return X.reshape(-1, 48 * 48)

# 1. Geometric Transformations (rotation, flip, shift)
def geometric_augmentation(X, Y):
    X_res = reshape_images(X)
    X_minority = X_res[Y == 0]
    Y_minority = Y[Y == 0]
    
    # Generate samples until the dataset is balanced
    while len(Y_minority) < len(Y[Y == 1]):
        for img in X_minority:
            if len(Y_minority) >= len(Y[Y == 1]):
                break
            # Apply random geometric transformations
            transformation = random.choice(['rotate', 'flip', 'shift'])
            if transformation == 'rotate':
                angle = random.randint(-30, 30)
                M = cv2.getRotationMatrix2D((24, 24), angle, 1.0)
                img_aug = cv2.warpAffine(img, M, (48, 48))
            elif transformation == 'flip':
                img_aug = cv2.flip(img, 1)
            elif transformation == 'shift':
                tx, ty = random.randint(-5, 5), random.randint(-5, 5)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                img_aug = cv2.warpAffine(img, M, (48, 48))
            X_minority = np.vstack([X_minority, [img_aug]])
            Y_minority = np.append(Y_minority, 0)

    X_balanced = np.vstack([X_res[Y == 1], X_minority])
    Y_balanced = np.concatenate([Y[Y == 1], Y_minority])
    
    return flatten_images(X_balanced), Y_balanced

# 2. Color Space Transformations (brightness, contrast, inversion)
def color_augmentation(X, Y):
    X_res = reshape_images(X)
    X_minority = X_res[Y == 0]
    Y_minority = Y[Y == 0]
    
    while len(Y_minority) < len(Y[Y == 1]):
        for img in X_minority:
            if len(Y_minority) >= len(Y[Y == 1]):
                break
            # Apply random color space transformations
            transformation = random.choice(['brightness', 'contrast', 'invert'])
            if transformation == 'brightness':
                factor = random.uniform(0.7, 1.3)
                img_aug = cv2.convertScaleAbs(img, alpha=factor, beta=0)
            elif transformation == 'contrast':
                factor = random.uniform(0.7, 1.3)
                img_aug = cv2.convertScaleAbs(img, alpha=1, beta=factor)
            elif transformation == 'invert':
                img_aug = cv2.bitwise_not(img)
            X_minority = np.vstack([X_minority, [img_aug]])
            Y_minority = np.append(Y_minority, 0)

    X_balanced = np.vstack([X_res[Y == 1], X_minority])
    Y_balanced = np.concatenate([Y[Y == 1], Y_minority])
    
    return flatten_images(X_balanced), Y_balanced

# 3. Noise Injection (Gaussian noise)
def noise_augmentation(X, Y):
    X_res = reshape_images(X)
    X_minority = X_res[Y == 0]
    Y_minority = Y[Y == 0]
    
    while len(Y_minority) < len(Y[Y == 1]):
        for img in X_minority:
            if len(Y_minority) >= len(Y[Y == 1]):
                break
            # Apply Gaussian noise
            noise = np.random.normal(0, 0.1, img.shape)
            img_aug = np.clip(img + noise * 255, 0, 255).astype(np.uint8)
            X_minority = np.vstack([X_minority, [img_aug]])
            Y_minority = np.append(Y_minority, 0)

    X_balanced = np.vstack([X_res[Y == 1], X_minority])
    Y_balanced = np.concatenate([Y[Y == 1], Y_minority])
    
    return flatten_images(X_balanced), Y_balanced

# 4. Combination of All
def combined_augmentation(X, Y):
    X_res = reshape_images(X)
    X_minority = X_res[Y == 0]
    Y_minority = Y[Y == 0]
    
    while len(Y_minority) < len(Y[Y == 1]):
        for img in X_minority:
            if len(Y_minority) >= len(Y[Y == 1]):
                break
            # Randomly choose a type of augmentation
            method = random.choice(['geometric', 'color', 'noise'])
            if method == 'geometric':
                # Apply random geometric transformation
                transformation = random.choice(['rotate', 'flip', 'shift'])
                if transformation == 'rotate':
                    angle = random.randint(-30, 30)
                    M = cv2.getRotationMatrix2D((24, 24), angle, 1.0)
                    img_aug = cv2.warpAffine(img, M, (48, 48))
                elif transformation == 'flip':
                    img_aug = cv2.flip(img, 1)
                elif transformation == 'shift':
                    tx, ty = random.randint(-5, 5), random.randint(-5, 5)
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    img_aug = cv2.warpAffine(img, M, (48, 48))
            elif method == 'color':
                # Apply random color space transformation
                transformation = random.choice(['brightness', 'contrast', 'invert'])
                if transformation == 'brightness':
                    factor = random.uniform(0.7, 1.3)
                    img_aug = cv2.convertScaleAbs(img, alpha=factor, beta=0)
                elif transformation == 'contrast':
                    factor = random.uniform(0.7, 1.3)
                    img_aug = cv2.convertScaleAbs(img, alpha=1, beta=factor)
                elif transformation == 'invert':
                    img_aug = cv2.bitwise_not(img)
            elif method == 'noise':
                # Apply Gaussian noise
                noise = np.random.normal(0, 0.1, img.shape)
                img_aug = np.clip(img + noise * 255, 0, 255).astype(np.uint8)
            
            X_minority = np.vstack([X_minority, [img_aug]])
            Y_minority = np.append(Y_minority, 0)

    X_balanced = np.vstack([X_res[Y == 1], X_minority])
    Y_balanced = np.concatenate([Y[Y == 1], Y_minority])
    
    return flatten_images(X_balanced), Y_balanced

# Test one of the functions (e.g., geometric augmentation)
X_geo_augmented, Y_geo_augmented = geometric_augmentation(X_train, Y_train)
X_geo_augmented.shape, np.unique(Y_geo_augmented, return_counts=True)
