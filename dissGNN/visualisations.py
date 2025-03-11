import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np


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
    # plt.show()


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

    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

    gdf_merged.plot(column="act", cmap="OrRd", legend=True, ax=axes[0])
    axes[0].set_title("Actual")

    gdf_merged.plot(column="pred", cmap="OrRd", legend=True, ax=axes[1])
    axes[1].set_title("Predicted")

    plt.savefig("output/map_visual/predictions/map_visual_population_counts.png", dpi=500)
    # plt.show()

def plot_residuals(pred, act, model):
    residuals = act - pred

    plt.figure(figsize=(8, 6))

    plt.scatter(pred, residuals, alpha=0.7, edgecolors='k')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.savefig("output/" + str(type(model).__name__) + "/residuals_plot.png", dpi=500)
    # plt.show()

def plot_graph_on_shapefile(shapefile_path_coarse, shapefile_path_fine, data, ad_level_coarse, ad_level_fine):

    # Load the shapefile
    gdf_coarse = gpd.read_file(shapefile_path_coarse)
    gdf_fine = gpd.read_file(shapefile_path_fine)

    # Ensure PCODEs match between graph data and shapefile
    gdf_coarse["ADM"+str(ad_level_coarse)+"_PCODE"] = gdf_coarse["ADM"+str(ad_level_coarse)+"_PCODE"].astype(str).str.strip()
    gdf_fine["ADM"+str(ad_level_fine)+"_PCODE"] = gdf_fine["ADM"+str(ad_level_fine)+"_PCODE"].astype(str).str.strip()
    data.admin_codes = pd.Series(data.admin_codes).astype(str).str.strip()

    admin_codes_coarse = set(data.admin_codes[data.admin_level.cpu().numpy() == ad_level_coarse])
    admin_codes_fine = set(data.admin_codes[data.admin_level.cpu().numpy() == ad_level_fine])

    # Merge node coordinates with shapefile data
    gdf_matched_coarse = gdf_coarse[gdf_coarse["ADM"+str(ad_level_coarse)+"_PCODE"].isin(admin_codes_coarse)].copy()
    gdf_matched_fine = gdf_fine[gdf_fine["ADM"+str(ad_level_fine)+"_PCODE"].isin(admin_codes_fine)].copy()
    gdf_matched_coarse["admin_level"] = "ADM2"
    gdf_matched_fine["admin_level"] = "ADM3"
    gdf_matched = pd.concat([gdf_matched_coarse, gdf_matched_fine])
    gdf_matched = gdf_matched.drop_duplicates()

    
    # Extract centroids as node coordinates
    gdf_matched["centroid"] = gdf_matched.geometry.centroid
    node_coords = np.array(list(zip(gdf_matched.centroid.x, gdf_matched.centroid.y)))

    # Create a graph using NetworkX
    edges = data.edge_index.cpu().numpy().T  # Convert to NumPy
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

     # Remove nodes that have no position
    valid_nodes = set(gdf_matched.index)
    edges_df = edges_df[edges_df["source"].isin(valid_nodes) & edges_df["target"].isin(valid_nodes)]

    G = nx.from_pandas_edgelist(edges_df, source="source", target="target", create_using=nx.Graph())


    # Convert node coordinates into a dictionary (for NetworkX)
    pos = {i: (node_coords[idx, 0], node_coords[idx, 1]) for idx, i in enumerate(valid_nodes)}

    # Ensure valid_nodes indices are within gdf_matched's index range
    valid_nodes = valid_nodes.intersection(gdf_matched.index)

    pos_coarse = gdf_matched[gdf_matched["admin_level"] == "ADM2"].apply(lambda row: pos[row.name] if row.name in pos else None, axis=1).dropna().to_dict()
    pos_fine = gdf_matched[gdf_matched["admin_level"] == "ADM3"].apply(lambda row: pos[row.name] if row.name in pos else None, axis=1).dropna().to_dict()

    pos_all = {**pos_coarse, **pos_fine}


    # Plot the shapefile as the base map
    fig, ax = plt.subplots(figsize=(12, 8))
    gdf_matched.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.6)

    G_coarse = G.subgraph(pos_coarse.keys())
    G_fine = G.subgraph(pos_fine.keys())

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="red", alpha=0.7) #Draw in edges first


    # Draw ADM3 nodes (green)
    nx.draw_networkx_nodes(G_fine, pos_fine, ax=ax, node_size=30, node_color="green", alpha=0.6)
    
    # Overlay the graph structure
    nx.draw_networkx_nodes(G_coarse, pos_coarse, ax=ax, node_size=50, node_color="blue", alpha=0.8)

    

    # # Add labels
    # for node, (x, y) in pos_coarse.items():
    #     ax.text(x, y, "ADM2", fontsize=9, color="black", ha="right", bbox=dict(facecolor="white", alpha=0.6))

    # for node, (x, y) in pos_fine.items():
    #     ax.text(x, y, "ADM3", fontsize=9, color="darkred", ha="left", bbox=dict(facecolor="white", alpha=0.6))

    # Formatting
    plt.title("Graph Structure Overlay on Shapefile")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("on")

    # Save and display
    plt.savefig("output/map_visual/admin_2_3/graph_on_shapefile.png", dpi=500)
    plt.show()
