import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def r_squared(y_true, y_pred):
    y_mean = np.mean(y_true)
    total_variance = np.sum((y_true - y_mean) ** 2)
    residual_variance = np.sum((y_true - y_pred) ** 2)
    return 1 - (residual_variance / total_variance)

def detect_outliers(data):
    median = np.median(data)
    deviation = np.abs(data - median)
    mad = np.median(deviation)
    return (data - median) / (mad / 0.6745) > 1.5

# Load datasets
x_train_path = "X_train.npy"
y_train_path = "y_train.npy"
x_test_path = "X_test.npy"

x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_test = np.load(x_test_path)

# Standardize features
scaler = StandardScaler()
x_train_standardized = scaler.fit_transform(x_train)

# Train Ridge regression model
ridge = Ridge(alpha=0.1)
ridge.fit(x_train_standardized, y_train)

# Predictions and error calculation
y_train_pred = ridge.predict(x_train_standardized)
print("Initial R^2: ", r_squared(y_train, y_train_pred))
prediction_errors = y_train - y_train_pred

# Plot error distribution
plt.figure(figsize=(10, 6))
plt.hist(prediction_errors, bins=50, edgecolor='black')
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
# plt.show()

# Identify and exclude outliers
outliers = detect_outliers(prediction_errors)
x_train_cleaned = x_train_standardized[~outliers]
y_train_cleaned = y_train[~outliers]

# Report number of outliers
print("Number of outliers removed: ", np.sum(outliers))

# Split cleaned data into training and validation sets
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train_cleaned, y_train_cleaned, test_size=0.2, random_state=42)

# Retrain model on cleaned data
ridge.fit(x_train_split, y_train_split)

# Validate model
y_val_pred = ridge.predict(x_val_split)
print("Improved R^2: ", r_squared(y_val_split, y_val_pred))
validation_errors = y_val_split - y_val_pred

# Plot validation error distribution
plt.figure(figsize=(10, 6))
plt.hist(validation_errors, bins=50, edgecolor='black')
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Validation Error Distribution")
plt.show()

# Standardize test data
x_test_standardized = scaler.transform(x_test)

# Predict on test data
y_test_pred = ridge.predict(x_test_standardized)

# Save predictions to file
np.save("y_pred.npy", y_test_pred)
