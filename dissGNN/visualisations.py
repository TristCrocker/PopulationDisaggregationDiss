import matplotlib.pyplot as plt
import networkx as nx


def plot_loss_val_curve(loss_arr, val_acc_arr, train_acc_arr, model_inst):
    plt.figure(figsize=(8, 5))
    x = range(len(loss_arr))
    # Plot training loss
    plt.plot(x, loss_arr, label="Train Loss", marker='o')

    # Plot validation MAE
    plt.plot(x, val_acc_arr, label="Val MAPE", marker='o')

    # Plot validation MAE
    plt.plot(x, train_acc_arr, label="Train MAPE", marker='o')

    plt.title("Training Loss and Validation MAPE over Epochs" + " (" + type(model_inst).__name__ + ")")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.tight_layout()

    # Save the figure to disk
    plt.savefig("output/" + type(model_inst).__name__ + "/train_loss_curve.png", dpi=300)
    plt.show()


def plot_graph_structure(data):
