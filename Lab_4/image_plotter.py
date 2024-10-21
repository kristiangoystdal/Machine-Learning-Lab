from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import matplotlib.pyplot as plt

from data_loader import load_data

Xtrain2_a, Xtrain2_b, Ytrain2_a, Ytrain2_b, Xtest2_a, Xtest2_b = load_data()

# Initialize the index
index = 0


# Function to update the image
def update_image():
    global index
    ax1.clear()
    ax2.clear()

    # Display Xtrain2_b (input image) in the left subplot
    ax1.imshow(Xtrain2_b[index].reshape(48, 48), cmap="gray")
    ax1.set_title("Input Image")

    # Display Ytrain2_b (segmentation mask) in the right subplot
    ax2.imshow(Ytrain2_b[index].reshape(48, 48), cmap="gray")
    num_ones = (Ytrain2_b[index] == 1).sum()
    num_zeros = (Ytrain2_b[index] == 0).sum()
    ax2.set_title(f"Number of 1s: {num_ones} | Number of 0s: {num_zeros}")

    canvas.draw()


# Function to handle key press events
def on_key(event):
    global index
    if event.keysym == "Right":
        index = (index + 1) % len(Ytrain2_b)
    elif event.keysym == "Left":
        index = (index - 1) % len(Ytrain2_b)
    elif event.keysym == "Escape":
        quit()
    update_image()


# Create the main window
root = tk.Tk()
root.title("Image Slideshow")

# Set the window to fullscreen
root.attributes("-fullscreen", True)

# Create a matplotlib figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2)  # Create two subplots, side by side

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
