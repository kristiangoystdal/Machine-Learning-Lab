import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt

# Load datasets
x_train_path = "u_train.npy"
y_train_path = "output_train.npy"
x_test_path = "u_test.npy"

x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_test = np.load(x_test_path)

# Model parameters
N_y = len(y_train)
N_x = len(x_train)


def phi(k, v_x, v_y, n, m, d):
    phi_vector = []
    # Use y_train or y_test_pred for the autoregressive part
    for i in range(1, n + 1):
        if k - i >= 0:
            phi_vector.append(v_y[k - i])
        else:
            phi_vector.append(0)  # Zero padding for the beginning
    # Use x_train or x_test for the exogenous part
    for i in range(m + 1):
        phi_vector.append(v_x[k - d - i])
    return phi_vector


def calculateXandY(n, m, d, p, N_y, y_train):
    X_train = []
    Y_train = []
    for i in range(p, N_y):
        X_train.append(phi(i, x_train, y_train, n, m, d))
        Y_train.append(y_train[i])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, Y_train, scaler


ridge = Ridge(alpha=0.1)
lasso = Lasso(alpha=0.1)


def trainModels(X_train, Y_train):
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, Y_train, test_size=0.2
    )

    ridge.fit(X_train_split, y_train_split)
    ridge_predictions = ridge.predict(X_val_split)

    lasso.fit(X_train_split, y_train_split)
    lasso_predictions = lasso.predict(X_val_split)

    return y_val_split, ridge_predictions, lasso_predictions


def calculateCoefficientOfDetermination(y_true, y_pred):
    y_mean = np.mean(y_true)
    total_variance = np.sum((y_true - y_mean) ** 2)
    residual_variance = np.sum((y_true - y_pred) ** 2)
    return 1 - (residual_variance / total_variance)


results = []
for n in range(1, 11):
    for m in range(1, 11):
        for d in range(1, 11):
            p = max(n, d + m)
            X_train, Y_train, scaler = calculateXandY(n, m, d, p, N_y, y_train)
            y_val_split, ridge_predictions, lasso_predictions = trainModels(
                X_train, Y_train
            )
            results.append(
                (
                    n,
                    m,
                    d,
                    p,
                    calculateCoefficientOfDetermination(y_val_split, ridge_predictions),
                    calculateCoefficientOfDetermination(y_val_split, lasso_predictions),
                )
            )

with open("results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["n", "m", "d", "p", "Ridge R^2", "Lasso R^2"])
    writer.writerows(results)

results.sort(key=lambda x: x[4], reverse=True)
print("Best ridge parameters:", results[0])

# Best ridge model
n = results[0][0]
m = results[0][1]
d = results[0][2]
p = results[0][3]

# Get the best model training
X_train, Y_train, scaler = calculateXandY(n, m, d, p, N_y, y_train)
ridge.fit(X_train, Y_train)

# Combine x_train and x_test
x_full = np.concatenate((x_train, x_test), axis=0)
N_full = len(x_full)

# Prepare for predictions on x_test
X_test = []
y_test_pred = []

# Initial predictions using y_train
for i in range(N_y, N_full):
    X_test.append(
        phi(i, x_full, np.concatenate((y_train, y_test_pred), axis=0), n, m, d)
    )
    X_test_scaled = scaler.transform([X_test[-1]])  # Use the same scaler
    y_test_pred.append(ridge.predict(X_test_scaled)[0])

# Save the predictions
np.save("output_test.npy", y_test_pred)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_train)), y_train, label="True Output", color="blue")
plt.plot(
    range(len(y_train), len(y_train) + len(y_test_pred)),
    y_test_pred,
    label="Ridge Predictions",
    color="red",
)
plt.title("ARX Model: Ridge Regression")
plt.legend()
plt.show()