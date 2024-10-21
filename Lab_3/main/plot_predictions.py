import matplotlib.pyplot as plt


def plot_images(x, y):
    fig, axes = plt.subplots(1, 5, figsize=(15, 12))

    for i, ax in enumerate(axes):
        ax.imshow(x[i].reshape(48, 48), cmap="gray")
        ax.set_title(f"Pred: {y[i]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_list(x):
    plt.scatter(range(len(x)), x)
    plt.xlabel("Iteration")
    plt.ylabel("F1 Score")
    plt.title("F1 Score over Iterations")
    plt.show()


def plot_average(x1, x2, x3, x4):
    args = [x1, x2, x3, x4]
    for i, x in enumerate(args, start=1):
        averages = [sum(values) / len(values) for values in zip(*x)]
        plt.figure()
        plt.scatter(
            range(len(averages)), averages, label=f"Average Values for {i} Lists"
        )
        plt.xlabel("Index")
        plt.ylabel("Average Value")
    plt.show()
