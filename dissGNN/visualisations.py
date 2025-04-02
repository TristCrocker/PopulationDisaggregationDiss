import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np

# Plot the structure of the input graph
def plot_graph_structure(data):
    # Convert edges to DF
    edges = data.edge_index.cpu().numpy().T
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    # Create a graph
    G = nx.from_pandas_edgelist(edges_df, source="source", target="target", create_using=nx.Graph())

    # Compute the number subgraphs
    num_subgraphs = nx.number_connected_components(G)
    print(f"Number of connected components (subgraphs): {num_subgraphs}")
    components = list(nx.connected_components(G))  #
    component_colors = {node: idx for idx, component in enumerate(components) for node in component}
    node_colors = [component_colors[node] for node in G.nodes()]

    # Position the nodes
    pos = nx.spring_layout(G, seed=42)

    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, cmap=plt.cm.Set1)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    plt.title(f"Graph Structure (Subgraphs: {num_subgraphs})")
    plt.axis("off")
    plt.savefig("output/graph_structure/graph_structure.png", dpi=500)

    return num_subgraphs

# Plot structure of country (Shapefile)
def plot_shape_file(shapefile_path, admin_level):
    # Read file
    gdf = gpd.read_file(shapefile_path)

    # Plot
    gdf.plot(figsize=(8, 6), edgecolor="black", alpha=0.5)
    plt.title("Admin " + str(admin_level) + " Layout")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("output/map_visual/admin_" + str(admin_level) + "/map_visual_population_counts.png", dpi=500)

# Plot predictions on the shapefile (Predicted vs actual)
def plot_shape_file_predictions(shapefile_path, pred, act, admin_level, data, model_inst):
    # Init setup
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.rename(columns={"ADM"+str(admin_level)+"_PCODE" : "ADM_PCODE"})
    mask_admin_level = (data.admin_level == admin_level)
    admin_codes = data.admin_codes
    df = pd.DataFrame({"ADM_PCODE" : admin_codes[mask_admin_level.numpy()], "act" : act, "pred" : pred})
    gdf_merged = gdf.merge(df, on="ADM_PCODE", how="left")

    # Produce plots
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

    # Font setup
    plt.rcParams.update({'font.size': 14})
    label_font = {'fontsize': 18, 'fontweight': 'bold'}
    title_font = {'fontsize': 18, 'fontweight': 'bold'}
    tick_fontsize = 16

    # Plot predictions on shapefile
    gdf_merged.plot(column="act", cmap="OrRd", legend=True, ax=axes[0])
    axes[0].set_title("Actual", **title_font)
    axes[0].set_xlabel("Longitude", fontsize=18, fontweight='bold')
    axes[0].set_ylabel("Latitude", fontsize=18, fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=tick_fontsize)

    gdf_merged.plot(column="pred", cmap="OrRd", legend=True, ax=axes[1], vmin=0, vmax=3500)
    axes[1].set_title("Predicted", **title_font)
    axes[1].set_xlabel("Longitude", fontsize=18, fontweight='bold')
    axes[1].set_ylabel("Latitude", fontsize=18, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=tick_fontsize)
    plt.savefig("output/" + type(model_inst).__name__ + "/mapped_predictions.png", dpi=500)

# Plot residuals
def plot_residuals(pred, act, model):
    # Calc residuals
    residuals = act - pred

    # Create plot
    plt.figure(figsize=(8, 6))

    # Font setup
    plt.rcParams.update({'font.size': 14})
    label_font = {'fontsize': 18, 'fontweight': 'bold'}
    title_font = {'fontsize': 18, 'fontweight': 'bold'}
    tick_fontsize = 16

    # Plot
    plt.scatter(pred, residuals, alpha=0.7, edgecolors='k')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.xlabel("Predicted Values", **label_font)
    plt.ylabel("Residuals", **label_font)
    plt.title("Residual Plot", **title_font)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig("output/" + str(type(model).__name__) + "/residuals_plot.png", dpi=500)

# Plot graph structure on the shapefile (validates input data)
def plot_graph_on_shapefile(shapefile_path_coarse, shapefile_path_fine, data, ad_level_coarse, ad_level_fine):
    # Load the shapefile
    gdf_coarse = gpd.read_file(shapefile_path_coarse)
    gdf_fine = gpd.read_file(shapefile_path_fine)

    # Init setup
    pcode_col_coarse = f"ADM{ad_level_coarse}_PCODE"
    pcode_col_fine = f"ADM{ad_level_fine}_PCODE"

    # Ensure correct naming
    gdf_coarse[pcode_col_coarse] = gdf_coarse[pcode_col_coarse].astype(str).str.strip()
    gdf_fine[pcode_col_fine] = gdf_fine[pcode_col_fine].astype(str).str.strip()
    data.admin_codes = pd.Series(data.admin_codes).astype(str).str.strip()

    # Partition into coarse and fine
    admin_codes_coarse = set(data.admin_codes[data.admin_level.cpu().numpy() == ad_level_coarse])
    admin_codes_fine = set(data.admin_codes[data.admin_level.cpu().numpy() == ad_level_fine])

    # Merge node coordinates with shapefile data
    gdf_matched_coarse = gdf_coarse[gdf_coarse[pcode_col_coarse].isin(admin_codes_coarse)].copy()
    gdf_matched_fine = gdf_fine[gdf_fine[pcode_col_fine].isin(admin_codes_fine)].copy()

    # Add col
    gdf_matched_coarse["admin_level"] = "ADM2"
    gdf_matched_fine["admin_level"] = "ADM3"

    # Extract node centres
    gdf_matched_coarse["centroid"] = gdf_matched_coarse.geometry.centroid
    gdf_matched_fine["centroid"] = gdf_matched_fine.geometry.centroid

    # Get coarse and fine positions
    pos_coarse = {row[pcode_col_coarse]: (row.centroid.x, row.centroid.y) for i, row in gdf_matched_coarse.iterrows()}
    pos_fine = {row[pcode_col_fine]: (row.centroid.x, row.centroid.y) for i, row in gdf_matched_fine.iterrows()}

    pos = {**pos_coarse, **pos_fine}

    # Create DF for edges
    edges = data.edge_index.cpu().numpy().T
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    node_index_to_pcode = {i: data.admin_codes[i] for i in range(len(data.admin_codes))}
    edges_df["source"] = edges_df["source"].map(node_index_to_pcode)
    edges_df["target"] = edges_df["target"].map(node_index_to_pcode)

    # Remove nodes that have been removed
    edges_df = edges_df[edges_df["source"].isin(pos) & edges_df["target"].isin(pos)]

    # Create graph
    G = nx.from_pandas_edgelist(edges_df, source="source", target="target", create_using=nx.Graph())

    # Plot the shapefile as base
    fig, ax = plt.subplots(figsize=(8, 12))

    # Font size setup
    plt.rcParams.update({'font.size': 14})
    label_font = {'fontsize': 18, 'fontweight': 'bold'}
    title_font = {'fontsize': 18, 'fontweight': 'bold'}
    tick_fontsize = 16

    # Plot shapefiles for coarse and fine
    gdf_matched_coarse.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.6)
    gdf_matched_fine.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.6)

    # Partition into subgraphs
    G_coarse = G.subgraph(pos_coarse.keys())
    G_fine = G.subgraph(pos_fine.keys())

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="red", alpha=0.7)

    # Draw ADM3 nodes (green)
    nx.draw_networkx_nodes(G_fine, pos_fine, ax=ax, node_size=30, node_color="green", alpha=0.6)

    # Draw ADM2 nodes (blue)
    nx.draw_networkx_nodes(G_coarse, pos_coarse, ax=ax, node_size=50, node_color="blue", alpha=0.8)

    # Plot settings
    plt.title("Graph Structure Overlay on Shapefile", **title_font)
    plt.xlabel("Longitude",**label_font)
    plt.ylabel("Latitude", **label_font)
    plt.axis("on")
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    plt.savefig("output/map_visual/admin_2_3/graph_on_shapefile.png", dpi=500)

# Plot admin level population density distributions
def plot_admin_dists(data):

    # Separate admin level labels
    admin2_labels = data.y[data.admin_level == 2].cpu().numpy()
    admin3_labels = data.y[data.admin_level == 3].cpu().numpy()

    # Plot
    plt.figure(figsize=(12,5))

    # Admin level 2 dist
    plt.subplot(1, 2, 1)
    plt.hist(admin2_labels, bins=50, alpha=0.7, label="Admin Level 2", color='blue')
    plt.xlabel("Population Density")
    plt.ylabel("Count")
    plt.title("Admin Level 2 Population Density Distribution")

    # Admin level 3 dist
    plt.subplot(1, 2, 2)
    plt.hist(admin3_labels, bins=50, alpha=0.7, label="Admin Level 3", color='red')
    plt.xlabel("Population Density")
    plt.ylabel("Count")
    plt.title("Admin Level 3 Population Density Distribution")
    plt.savefig("output/admin_dists.png", dpi=500)
 

# Plot covariates dist on shapefile
def plot_shape_file_covariates(shapefile_path_coarse, shapefile_path_fine, df_data, covariate_col):
    # Read shapefile
    # Coarse
    gdf_coarse = gpd.read_file(shapefile_path_coarse)
    gdf_coarse = gdf_coarse.rename(columns={"ADM"+str(2)+"_PCODE" : "ADM_PCODE"})

    # Fine
    gdf_fine = gpd.read_file(shapefile_path_fine)
    gdf_fine = gdf_fine.rename(columns={"ADM"+str(3)+"_PCODE" : "ADM_PCODE"})

    # Create masks for levels
    mask_admin_level_coarse = (df_data["admin_level"] == 2)
    mask_admin_level_fine = (df_data["admin_level"] == 3)

    # Produce coarse DF
    df_coarse = pd.DataFrame({
        "ADM_PCODE": df_data["ADM_PCODE"][mask_admin_level_coarse].to_numpy(),
        "covariate_value": df_data[covariate_col][mask_admin_level_coarse].to_numpy()})
    
    # Produce fine DF
    df_fine = pd.DataFrame({
        "ADM_PCODE": df_data["ADM_PCODE"][mask_admin_level_fine].to_numpy(),
        "covariate_value": df_data[covariate_col][mask_admin_level_fine].to_numpy()})

    # Merge with shapefiles
    gdf_merged_coarse = gdf_coarse.merge(df_coarse, on="ADM_PCODE", how="left")
    gdf_merged_fine = gdf_fine.merge(df_fine, on="ADM_PCODE", how="left")

    # Plot
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

    # Check covariates values all numeric
    if pd.api.types.is_numeric_dtype(gdf_merged_coarse["covariate_value"]):
        # Get range
        vmin = min(gdf_merged_coarse["covariate_value"].min(), gdf_merged_fine["covariate_value"].min())
        vmax = max(gdf_merged_coarse["covariate_value"].max(), gdf_merged_fine["covariate_value"].max())

        # Coarse plot
        gdf_merged_coarse.plot(column="covariate_value", cmap="OrRd", legend=True, ax=axes[0], edgecolor="black", vmin = vmin, vmax = vmax)
        axes[0].set_title("Admin Level 2 - " + covariate_col)

        # Fine plot
        gdf_merged_fine.plot(column="covariate_value", cmap="OrRd", legend=True, ax=axes[1], edgecolor="black", vmin = vmin, vmax = vmax)
        axes[1].set_title("Admin Level 3 - " + covariate_col)

    else:
        # No range found as not all numeric
        # Coarse plot
        gdf_merged_coarse.plot(column="covariate_value", cmap="OrRd", legend=True, ax=axes[0], edgecolor="black")
        axes[0].set_title("Admin Level 2 - " + covariate_col)

        # Fine plot
        gdf_merged_fine.plot(column="covariate_value", cmap="OrRd", legend=True, ax=axes[1], edgecolor="black")
        axes[1].set_title("Admin Level 3 - " + covariate_col)

    # Cleaning for save
    covariate_col = covariate_col.replace("/", "")
    plt.savefig("output/map_visual/covariates/map_visual_covariate_" + covariate_col + ".png", dpi=500)

# Plot loss curve
def plot_acc_loss_over_epochs(model_losses, model_inst_arr):
    # Get number epochs
    epochs = range(1, len(model_losses[0]) + 1)

    # Plot
    fig, axs = plt.subplots(1, 1, figsize=(14, 6), sharex=True)

    # Font setup
    plt.rcParams.update({'font.size': 14})
    label_font = {'fontsize': 18, 'fontweight': 'bold'}
    title_font = {'fontsize': 18, 'fontweight': 'bold'}
    tick_fontsize = 16

    # Loop over loss array (Contains loss for a few models)
    for i, losses in enumerate(model_losses):
        # Plot losses
        axs.plot(epochs, losses, label=f'Model Loss ({type(model_inst_arr[i]).__name__})')

    # Plot setup
    axs.set_ylabel('Loss', **label_font)
    axs.set_title('Loss Over Epochs', **title_font)
    axs.set_xlabel('Epochs', **label_font)
    axs.legend()
    axs.tick_params(axis='both', labelsize=tick_fontsize)
    axs.grid(True)
    plt.tight_layout()

    # Check if only 1 model loss provided and save appropriately
    if len(model_losses) == 1:
        plt.savefig("output/" + type(model_inst_arr[0]).__name__ + "/train_loss_curve.png", dpi=500)
    else:
        plt.savefig("output/all_models/train_loss_curve.png", dpi=500)