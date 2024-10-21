from sklearn.feature_extraction import image
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data

# Load the data
Xtrain2_a, Xtrain2_b, Ytrain2_a, Ytrain2_b, Xtest2_a, Xtest2_b = load_data()

# Reshape Xtrain2_a to patches of 7x7
Xtrain2_a_reshaped = Xtrain2_a.reshape(-1, 7, 7)  # Reshaping back to 7x7 patches
print(f"Xtrain2_a Patch shape after reshaping: {Xtrain2_a_reshaped.shape}")

# Suppose the original image size is 48x48
image_size = (48, 48)

# Reconstruct the X image from patches
reconstructed_image = image.reconstruct_from_patches_2d(
    patches=Xtrain2_a_reshaped, image_size=image_size
)

# Reshape Ytrain2_a to patches of 7x7
Ytrain2_a_reshaped = Ytrain2_a.reshape(-1, 7, 7)
print(f"Ytrain2_a Patch shape after reshaping: {Ytrain2_a_reshaped.shape}")

# Now reconstruct the Y image from patches (0 or 1 binary values)
reconstructed_y_image = image.reconstruct_from_patches_2d(
    patches=Ytrain2_a_reshaped, image_size=image_size
)

reconstructed_y_image = np.round(reconstructed_y_image)  # Round to 0 or 1

# Check the reconstructed image shapes
print(f"Reconstructed X image shape: {reconstructed_image.shape}")
print(f"Reconstructed Y image shape: {reconstructed_y_image.shape}")

# Visualize the reconstructed X image
plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed X Image")
plt.show()

# # Visualize the binary reconstructed Y image (0 for background, 1 for crater)
# plt.imshow(reconstructed_y_image, cmap="gray", vmin=0, vmax=1)
# plt.title("Reconstructed Y Image (Binary Mask)")
# plt.show()
