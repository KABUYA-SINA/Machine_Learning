import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    # Create a new figure with a fixed size
    plt.figure(figsize=(5, 4))

    # Plot the confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    # Set plot title
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()

    # Image saving (useful for Flask / portfolio)
    if save_path:
        plt.savefig(save_path)

    # Display the plot
    plt.show()

    # Close the figure to free memory
    plt.close()