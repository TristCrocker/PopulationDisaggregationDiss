import geopandas as gpd
import networkx as nx
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import visualisations
from torch_geometric.utils import get_laplacian
import torch_geometric.utils as pyg_utils

# Convert a shapefile to edges on same admin level
def convert_shapefile_to_graph(shape_path, admin_level):
    shape = gpd.read_file(shape_path)
    shape = shape[shape.geometry.notnull()]

    # Dynamic check level of shapefile
    if admin_level == 3:
        ad_code = 'ADM3_PCODE'
    elif admin_level == 2:
        ad_code = 'ADM2_PCODE'
    elif admin_level == 1:
        ad_code = 'ADM1_PCODE'

    #Identify neighbours and create edges
    edges = []
    for i, district_1 in shape.iterrows():
        for j, district_2 in shape.iterrows():
            if i != j and district_1.geometry.touches(district_2.geometry):
                edges.append((district_1[ad_code], district_2[ad_code]))
                edges.append((district_2[ad_code], district_1[ad_code]))

    return pd.DataFrame(edges, columns=["source", "target"])

# Convert mappings file to inter-admin level edges
def convert_level_mappings_to_edges(mappings_path):
    mappings = pd.read_csv(mappings_path)
    edges = mappings[['ADM2_PCODE', 'ADM3_PCODE']]
    edges = edges.rename(columns={'ADM2_PCODE': 'source', 'ADM3_PCODE': 'target'})
    return edges

# Main load data function
def load_data(shape_path_coarse, shape_path_fine, features_path_coarse, features_path_fine, mappings_path, admin_level_coarse, admin_level_fine):

    #  -------- Join all edge dataframes which contains edges between admin 2 areas, admin 3 areas and edges between admin 2 and 3 areas --------
    # Coarse edges
    edges_coarse = convert_shapefile_to_graph(shape_path_coarse, admin_level_coarse)
    edges_coarse["type"] = "coarse"

    # Fine edges
    edges_fine = convert_shapefile_to_graph(shape_path_fine, admin_level_fine)
    edges_fine["type"] = "fine"

    # Mapping edges
    edges_mappings = convert_level_mappings_to_edges(mappings_path)
    edges_mappings["type"] = "mappings"

    # All edges combined
    edges = pd.concat([edges_coarse, edges_mappings])
    
    # -------- Add weights for edges --------
    weights = {"coarse" : 0.5, "fine" : 0.2, "mappings" : 3}
    edges["weights_init"] = edges["type"].map(weights)
    edges["weights_init"] = edges["weights_init"] / edges["weights_init"].max()

    # -------- Load node features for coarse admin and fine admin level and rename to generalize for concat --------
    # Coarse admin level
    col_coarse_pcode = "ADM" + str(admin_level_coarse) + "_PCODE"   
    col_coarse_pt = "ADM" + str(admin_level_coarse) + "_PT"
    node_features_coarse = pd.read_csv(features_path_coarse)
    node_features_coarse["admin_level"] = admin_level_coarse
    node_features_coarse = node_features_coarse.rename(columns={col_coarse_pcode : "ADM_PCODE", col_coarse_pt : "ADM_PT"})
    
    # Fine admin level
    col_fine_pcode = "ADM" + str(admin_level_fine) + "_PCODE"
    col_fine_pt = "ADM" + str(admin_level_fine) + "_PT"
    node_features_fine = pd.read_csv(features_path_fine)
    node_features_fine["admin_level"] = admin_level_fine
    node_features_fine = node_features_fine.rename(columns={col_fine_pcode : "ADM_PCODE", col_fine_pt : "ADM_PT"})

    # Combine node features for coarse and fine
    node_features = pd.concat([node_features_coarse, node_features_fine])
    
    # -------- Preprocess data numerically --------
    x_features, y_feature, edges_final, node_features = process_feature_data(node_features, edges)

    # Create covariates tensor
    x = torch.tensor(x_features, dtype=torch.float)

    # Create edges tensor
    edge_index = torch.tensor(edges_final[["source_num", "target_num"]].values.T, dtype=torch.long)

    #Create edge weights tensor
    edge_weights = torch.tensor(edges_final["weights_init"].values, dtype=torch.float)
    
    #Produce graph structure
    data = Data(x=x, edge_index=edge_index)
    data.edge_weight = edge_weights
    data.y = torch.tensor(y_feature.values, dtype=torch.float)

    # Store admin level and codes in data structure
    admin_level_vals = node_features["admin_level"].values
    data.admin_level = torch.tensor(admin_level_vals, dtype=torch.long)
    data.admin_codes = np.array(node_features.index.values.tolist())
    
    # --------- Train/val/test split ---------
    data.train_mask = (data.admin_level == 2) & (torch.rand(data.num_nodes) < 0.9) # Train (Admin level 2)
    data.val_mask = (data.admin_level == 2) & ~ data.train_mask # Val (Admin level 2)
    data.test_mask = (data.admin_level == 3) # Test (Admin level 3)

    return data

# Function to preprocess features and target y
def process_feature_data(node_features, edges):

    # --------- Setup data for preprocessing ---------
    node_features = node_features.copy()
    node_features["ADM_PCODE"] = node_features["ADM_PCODE"].astype(str)

    # --------- NaN value removal ---------
    # Store nodes with NaNs
    nodes_with_nans = node_features[node_features.isna().any(axis=1)]

    # Store pcodes of NaN nodes
    removed_pcodes = set(nodes_with_nans["ADM_PCODE"])  

    # Get admin level 2 NaN pcodes
    admin2_nan_pcodes = set(node_features[(node_features["admin_level"] == 2) & (node_features["ADM_PCODE"].isin(removed_pcodes))]["ADM_PCODE"])

    # Find corresponding admin level 3 nodes of removed admin level 2 nodes 
    admin3_remove = node_features[
        (node_features["admin_level"] == 3) &
        (node_features["ADM_PCODE"].str.startswith(tuple(admin2_nan_pcodes)))
    ]

    # Add these admin 3 nodes to the removal list
    removed_pcodes.update(admin3_remove["ADM_PCODE"])
    
    # Remove nodes from node_features which have been removed
    node_features = node_features[~node_features["ADM_PCODE"].isin(removed_pcodes)]

    # Remove edges which are connected to removed nodes
    edges = edges[edges["source"].isin(node_features["ADM_PCODE"]) & edges["target"].isin(node_features["ADM_PCODE"])].copy()
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    node_features.set_index("ADM_PCODE", inplace=True)  # Set pcode as index

    #  -------------- Preprocess certain things --------------
    #Produce population density
    node_features["population_density"] = node_features["T_TL"] / node_features["district_area"] 
    # Remove zero-population areas
    node_features = node_features[node_features["population_density"] > 0]  

    # Create target variable
    y_feature = node_features["population_density"]

    # Create x features
    x_unscaled = node_features.drop(columns=["population_density", "ADM_PT", "log_population", "T_TL"])

    #Convert certain variables into densities
    x_unscaled[['building_count', 'building_area', 'osm_traffic', 'osm_transport', 'osm_places', 'osm_pofw', 'osm_pois', 'osm_railways', 'osm_roads']] = x_unscaled[['building_count', 'building_area', 'osm_traffic', 'osm_transport', 'osm_places', 'osm_pofw', 
                  'osm_pois', 'osm_railways', 'osm_roads']].div(x_unscaled['district_area'], axis=0)

    ad_levels = x_unscaled["admin_level"].copy() # For visualisation below
    x_unscaled = x_unscaled.drop(columns=['admin_level'])

    # Produce log (1 + density) target variable
    y_feature_final = np.log1p(1 + y_feature)
    
    # --------- Visualisation of covariates --------------
    # original_index = x_unscaled.index  
    
    # # Standardize x (features)
    # scaler_vis = MinMaxScaler()
    # x_scaled = scaler_vis.fit_transform(x_unscaled)
    # # x_scaled = x_unscaled.values
    # # Create DataFrame with original index
    # x_features_cols_vis = pd.DataFrame(x_scaled, columns=x_unscaled.columns, index=original_index)
    # x_features_cols_vis['ADM_PCODE'] = original_index  
    # x_features_cols_vis["admin_level"] = ad_levels.reindex(original_index)  # Realign

    #Visualise covariates
    # for col in x_features_cols_vis.columns:
    #     visualisations.plot_shape_file_covariates("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", x_features_cols_vis, str(col))
    # -------------------------------------------------

    # Scale covariates
    scaler = StandardScaler()
    x_features = scaler.fit_transform(x_unscaled)

    #Convert edges to indexes
    area_to_index = {code: idx for idx, code in enumerate(node_features.index)}
    edges = edges[edges["source"].isin(area_to_index) & edges["target"].isin(area_to_index)].copy()
    edges["source_num"] = edges["source"].map(area_to_index)
    edges["target_num"] = edges["target"].map(area_to_index)
    
    return x_features, y_feature_final, edges, node_features


