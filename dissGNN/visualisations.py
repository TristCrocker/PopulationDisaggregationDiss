import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import geopandas as gpd


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


def plot_shape_file(shapefile_path, admin_level):
    gdf = gpd.read_file(shapefile_path)


    gdf.plot(figsize=(8, 6), edgecolor="black", alpha=0.5)

    plt.title("Admin " + str(admin_level) + " Layout")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("output/map_visual/admin_" + str(admin_level) + "/map_visual_population_counts.png", dpi=500)
    plt.show()

def plot_shape_file_predictions(shapefile_path, pred, act, admin_level, data):
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.rename(columns={"ADM"+str(admin_level)+"_PCODE" : "ADM_PCODE"})
    mask_admin_level = (data.admin_level == admin_level)
    admin_codes = data.admin_codes
    df = pd.DataFrame({"ADM_PCODE" : admin_codes[mask_admin_level.numpy()], "act" : act, "pred" : pred})

    gdf_merged = gdf.merge(df, on="ADM_PCODE", how="left")

    # Step 4: Plot side by side
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

    gdf_merged.plot(column="act", cmap="OrRd", legend=True, ax=axes[0])
    axes[0].set_title("Actual")

    gdf_merged.plot(column="pred", cmap="OrRd", legend=True, ax=axes[1])
    axes[1].set_title("Predicted")

    plt.savefig("output/map_visual/predictions/map_visual_population_counts.png", dpi=500)
    plt.show()
