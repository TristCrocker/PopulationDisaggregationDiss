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
    # plt.show()

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

    pcode_col_coarse = f"ADM{ad_level_coarse}_PCODE"
    pcode_col_fine = f"ADM{ad_level_fine}_PCODE"

    # Ensure correct naming
    gdf_coarse[pcode_col_coarse] = gdf_coarse[pcode_col_coarse].astype(str).str.strip()
    gdf_fine[pcode_col_fine] = gdf_fine[pcode_col_fine].astype(str).str.strip()
    data.admin_codes = pd.Series(data.admin_codes).astype(str).str.strip()

    admin_codes_coarse = set(data.admin_codes[data.admin_level.cpu().numpy() == ad_level_coarse])
    admin_codes_fine = set(data.admin_codes[data.admin_level.cpu().numpy() == ad_level_fine])


    # Merge node coordinates with shapefile data
    gdf_matched_coarse = gdf_coarse[gdf_coarse[pcode_col_coarse].isin(admin_codes_coarse)].copy()
    gdf_matched_fine = gdf_fine[gdf_fine[pcode_col_fine].isin(admin_codes_fine)].copy()

    gdf_matched_coarse["admin_level"] = "ADM2"
    gdf_matched_fine["admin_level"] = "ADM3"

    # Extract centroids as node coordinates
    gdf_matched_coarse["centroid"] = gdf_matched_coarse.geometry.centroid
    gdf_matched_fine["centroid"] = gdf_matched_fine.geometry.centroid

    pos_coarse = {row[pcode_col_coarse]: (row.centroid.x, row.centroid.y) for _, row in gdf_matched_coarse.iterrows()}
    pos_fine = {row[pcode_col_fine]: (row.centroid.x, row.centroid.y) for _, row in gdf_matched_fine.iterrows()}

    pos = {**pos_coarse, **pos_fine}

    # Create a graph using NetworkX
    edges = data.edge_index.cpu().numpy().T  # Convert to NumPy
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    node_index_to_pcode = {i: data.admin_codes[i] for i in range(len(data.admin_codes))}
    edges_df["source"] = edges_df["source"].map(node_index_to_pcode)
    edges_df["target"] = edges_df["target"].map(node_index_to_pcode)

     # Remove nodes that have no position
    edges_df = edges_df[edges_df["source"].isin(pos) & edges_df["target"].isin(pos)]

    G = nx.from_pandas_edgelist(edges_df, source="source", target="target", create_using=nx.Graph())

    # Plot the shapefile as the base map
    fig, ax = plt.subplots(figsize=(12, 8))
    gdf_matched_coarse.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.6)
    gdf_matched_fine.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.6)

    G_coarse = G.subgraph(pos_coarse.keys())
    G_fine = G.subgraph(pos_fine.keys())

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="red", alpha=0.7) #Draw in edges first


    # Draw ADM3 nodes (green)
    nx.draw_networkx_nodes(G_fine, pos_fine, ax=ax, node_size=30, node_color="green", alpha=0.6)

    # Overlay the graph structure
    nx.draw_networkx_nodes(G_coarse, pos_coarse, ax=ax, node_size=50, node_color="blue", alpha=0.8)
    # Formatting
    plt.title("Graph Structure Overlay on Shapefile")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("on")

    # Save and display
    plt.savefig("output/map_visual/admin_2_3/graph_on_shapefile.png", dpi=500)
    plt.show()


def plot_admin_dists(data):


    # Separate labels for admin level 2 and 3
    admin2_labels = data.y[data.admin_level == 2].cpu().numpy()
    admin3_labels = data.y[data.admin_level == 3].cpu().numpy()

    # Plot histograms
    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    plt.hist(admin2_labels, bins=50, alpha=0.7, label="Admin Level 2", color='blue')
    plt.xlabel("Population Density")
    plt.ylabel("Count")
    plt.title("Admin Level 2 Population Density Distribution")

    plt.subplot(1, 2, 2)
    plt.hist(admin3_labels, bins=50, alpha=0.7, label="Admin Level 3", color='red')
    plt.xlabel("Population Density")
    plt.ylabel("Count")
    plt.title("Admin Level 3 Population Density Distribution")
    plt.savefig("output/admin_dists.png", dpi=500)
    # plt.show()    


def plot_shape_file_covariates(shapefile_path_coarse, shapefile_path_fine, df_data, covariate_col):
    gdf_coarse = gpd.read_file(shapefile_path_coarse)
    gdf_coarse = gdf_coarse.rename(columns={"ADM"+str(2)+"_PCODE" : "ADM_PCODE"})

    gdf_fine = gpd.read_file(shapefile_path_fine)
    gdf_fine = gdf_fine.rename(columns={"ADM"+str(3)+"_PCODE" : "ADM_PCODE"})

    mask_admin_level_coarse = (df_data["admin_level"] == 2)
    mask_admin_level_fine = (df_data["admin_level"] == 3)
    df_coarse = pd.DataFrame({
        "ADM_PCODE": df_data["ADM_PCODE"][mask_admin_level_coarse].to_numpy(),
        "covariate_value": df_data[covariate_col][mask_admin_level_coarse].to_numpy()})
    
    df_fine = pd.DataFrame({
        "ADM_PCODE": df_data["ADM_PCODE"][mask_admin_level_fine].to_numpy(),
        "covariate_value": df_data[covariate_col][mask_admin_level_fine].to_numpy()})

    gdf_merged_coarse = gdf_coarse.merge(df_coarse, on="ADM_PCODE", how="left")
    gdf_merged_fine = gdf_fine.merge(df_fine, on="ADM_PCODE", how="left")

    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

    gdf_merged_coarse.plot(column="covariate_value", cmap="OrRd", legend=True, ax=axes[0], edgecolor="black")
    axes[0].set_title("Admin Level 2 - " + covariate_col)

    gdf_merged_fine.plot(column="covariate_value", cmap="OrRd", legend=True, ax=axes[1], edgecolor="black")
    axes[1].set_title("Admin Level 3 - " + covariate_col)

    covariate_col = covariate_col.replace("/", "")

    plt.savefig("output/map_visual/covariates/map_visual_covariate_" + covariate_col + ".png", dpi=500)
    # plt.show()