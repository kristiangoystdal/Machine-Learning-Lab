import numpy as np
import matplotlib.pyplot as plt

X_test = np.load("Xtest2_b.npy")
Y_test = np.load("Ytest.npy")
Y_test_bin = np.load("Ytest_binary.npy")

X_test = X_test.reshape((-1, 48, 48, 1))
Y_test = Y_test.reshape((-1, 48, 48, 1))
Y_test_bin = Y_test_bin.reshape((-1, 48, 48, 1))



def plotandshow(A):
    fig, axes = plt.subplots(3, 5, figsize=(14, 8))
    for i in range(5):
        # Display the test image
        axes[0, i].imshow(X_test[i+(5*A)].squeeze(), cmap='gray')
        axes[0, i].set_title(f"Input Image {i+1}")
        axes[0, i].axis('off')
        
        # Display the prediction
        axes[1, i].imshow(Y_test[i+(5*A)].squeeze(), cmap='gray')
        axes[1, i].set_title(f"Prediction {i+1}")
        axes[1, i].axis('off')

        axes[2, i].imshow(Y_test_bin[i+(5*A)].squeeze(), cmap='gray')
        axes[2, i].set_title(f"Prediction (binary) {i+1}")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()



for j in range((X_test.size//5)):
    plotandshow(j)