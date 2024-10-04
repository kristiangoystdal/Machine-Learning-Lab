# Group 13

import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def calculateCoefficientOfDetermination(y_true, y_pred):
    y_mean = np.mean(y_true)
    total_variance = np.sum((y_true - y_mean) ** 2)
    residual_variance = np.sum((y_true - y_pred) ** 2)
    return 1 - (residual_variance / total_variance)


def detectOutliers(data):
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

# Train Lasso regression model
lasso = Lasso(alpha=0.1)
lasso.fit(x_train_standardized, y_train)

# Predictions and error calculation
y_train_pred_ridge = ridge.predict(x_train_standardized)
print(
    "Initial CoD (ridge): ",
    calculateCoefficientOfDetermination(y_train, y_train_pred_ridge),
)
prediction_errors_ridge = y_train - y_train_pred_ridge

y_train_pred_lasso = lasso.predict(x_train_standardized)
print(
    "Initial CoD (lasso): ",
    calculateCoefficientOfDetermination(y_train, y_train_pred_lasso),
)
prediction_errors_lasso = y_train - y_train_pred_lasso

# Identify and exclude outliers
outliers_ridge = detectOutliers(prediction_errors_ridge)
x_train_cleaned_ridge = x_train_standardized[~outliers_ridge]
y_train_cleaned_ridge = y_train[~outliers_ridge]

outliers_lasso = detectOutliers(prediction_errors_lasso)
x_train_cleaned_lasso = x_train_standardized[~outliers_lasso]
y_train_cleaned_lasso = y_train[~outliers_lasso]

# Print number of outliers that were removed
print("Number of outliers removed (ridge): ", np.sum(outliers_ridge))

print("Number of outliers removed (lasso): ", np.sum(outliers_lasso))

# Split cleaned data into training and validation sets
x_train_split_ridge, x_val_split_ridge, y_train_split_ridge, y_val_split_ridge = (
    train_test_split(
        x_train_cleaned_ridge, y_train_cleaned_ridge, test_size=0.2, random_state=42
    )
)

x_train_split_lasso, x_val_split_lasso, y_train_split_lasso, y_val_split_lasso = (
    train_test_split(
        x_train_cleaned_lasso, y_train_cleaned_lasso, test_size=0.2, random_state=42
    )
)

# Retrain model on cleaned data
ridge.fit(x_train_split_ridge, y_train_split_ridge)

lasso.fit(x_train_split_lasso, y_train_split_lasso)

# Validate model
y_val_pred_ridge = ridge.predict(x_val_split_ridge)
print(
    "Improved CoD (ridge): ",
    calculateCoefficientOfDetermination(y_val_split_ridge, y_val_pred_ridge),
)
validation_errors = y_val_split_ridge - y_val_pred_ridge

y_val_pred_lasso = lasso.predict(x_val_split_lasso)
print(
    "Improved CoD (lasso): ",
    calculateCoefficientOfDetermination(y_val_split_lasso, y_val_pred_lasso),
)
validation_errors = y_val_split_lasso - y_val_pred_lasso

### Test data predictions using ridge model based on coefficients of determination from validation data ###

# Standardize test data
x_test_standardized = scaler.transform(x_test)

# Predict on test data
y_test_pred = ridge.predict(x_test_standardized)

# Save predictions to file
np.save("regression_test_pred.npy", y_test_pred)
