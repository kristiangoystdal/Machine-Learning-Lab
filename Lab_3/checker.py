import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import matplotlib.pyplot as plt

# Load the data
X_test = np.load("Xtest1.npy")
Y_test = np.load("Ytest1_pred.npy")

print(X_test.shape, Y_test.shape)

# Initialize the index
index = 0


# Function to update the image
def update_image():
    global index
    ax.clear()
    ax.imshow(
        X_test[index].reshape(48, 48), cmap="gray"
    )  # Assuming the image size is 48x48
    ax.set_title(f"Label: {Y_test[index]}")
    canvas.draw()


# Function to handle key press events
def on_key(event):
    global index
    if event.keysym == "Right":
        index = (index + 1) % len(X_test)
    elif event.keysym == "Left":
        index = (index - 1) % len(X_test)
    update_image()


# Create the main window
root = tk.Tk()
root.title("Image Slideshow")

# Set the window to fullscreen
root.attributes("-fullscreen", True)

# Create a matplotlib figure and axis
fig, ax = plt.subplots()

# Create a canvas to display the figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Bind the arrow keys to the on_key function
root.bind("<Right>", on_key)
root.bind("<Left>", on_key)

# Display the first image
update_image()

# Start the Tkinter event loop
root.mainloop()
