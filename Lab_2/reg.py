import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load datasets
x_train_path = "u_train.npy"
y_train_path = "output_train.npy"
x_test_path = "u_test.npy"

# Replace with actual loading if the .npy files are present
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_test = np.load(x_test_path)

# Model parameters
n = 3  # Autoregressive order
m = 2  # Exogenous order
d = 1  # Delay

# Prepare the regressor matrix for the training data
N_train = len(x_train)
X_train = np.zeros((N_train - n, n + m))
y_train_reg = y_train[n:]

for k in range(n, N_train):
    X_train[k - n, :n] = y_train[k - n : k][::-1]  # Past y values
    X_train[k - n, n:] = x_train[k - d : k - d - m : -1]  # Past u values

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train/test split from the training data
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train_reg, test_size=0.2, random_state=42
)

# Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_split, y_train_split)
ridge_predictions = ridge.predict(X_val_split)

# Lasso regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_split, y_train_split)
lasso_predictions = lasso.predict(X_val_split)

# Plot the results for comparison
plt.figure(figsize=(10, 6))
plt.plot(y_val_split, label="True Output", color="blue")
plt.plot(ridge_predictions, label="Ridge Predictions", color="red", linestyle="dashed")
plt.plot(
    lasso_predictions, label="Lasso Predictions", color="green", linestyle="dashed"
)
plt.title("ARX Model: Ridge and Lasso Regression")
plt.legend()
plt.show()

# Prepare the regressor matrix for the test data
N_test = len(x_test)
X_test = np.zeros((N_test - n, n + m))
y_test_pred = np.zeros(N_test)

# Initialize y_test_pred for predictions with known initial values
y_test_pred[:n] = y_train[-n:]

# Generate predictions for test data
for k in range(n, N_test):
    X_test[k - n, :n] = y_test_pred[k - n : k][::-1]  # Past predicted y values
    X_test[k - n, n:] = x_test[k - d : k - d - m : -1]  # Past u values

X_test = scaler.transform(X_test)
y_test_pred[n:] = ridge.predict(X_test)  # Using Ridge model for test predictions

# Output final predictions for test set
y_test_submission = y_test_pred[-400:]  # Last 400 samples

print(y_test_submission)
