import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
# import geopandas as pd


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


def plot_graph_structure(data):


    edges = data.edge_index.cpu().numpy().T
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    G = nx.from_pandas_edgelist(edges_df, source="source", target="target", create_using=nx.Graph())

    pos = nx.spring_layout(G, seed=42)  
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    # nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Topological View of the Graph")
    plt.axis("off")
    plt.savefig("output/graph_structure/graph_structure.png", dpi=500)
    plt.show()


# def plot_shape_file(shapefile_path):
