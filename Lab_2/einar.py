import matplotlib.pyplot as plt
import numpy as np

u_train_path = "u_train.npy"
y_train_path = "output_train.npy"
u_test_path = "u_test.npy"

u_train = np.load(u_train_path)
y_train = np.load(y_train_path)
u_test = np.load(u_test_path)

plt.plot(u_train)
plt.show()