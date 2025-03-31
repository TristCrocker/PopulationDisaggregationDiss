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


# Function to convert a shapefile to represent edges of graph for specific country
def convert_shapefile_to_graph(shape_path, admin_level):
    shape = gpd.read_file(shape_path)
    shape = shape[shape.geometry.notnull()]

    if admin_level == 3:
        ad_code = 'ADM3_PCODE'
    elif admin_level == 2:
        ad_code = 'ADM2_PCODE'
    elif admin_level == 1:
        ad_code = 'ADM1_PCODE'

    #Identify neighbours
    edges = []
    for i, district_1 in shape.iterrows():
        for j, district_2 in shape.iterrows():
            if i != j and district_1.geometry.touches(district_2.geometry):
                edges.append((district_1[ad_code], district_2[ad_code]))
                edges.append((district_2[ad_code], district_1[ad_code]))

    return pd.DataFrame(edges, columns=["source", "target"])

def convert_level_mappings_to_edges(mappings_path):
    
    mappings = pd.read_csv(mappings_path)
    
    # Print duplicate admin level 3
    # duplicate_values = mappings[mappings.duplicated(subset=["ADM3_PCODE"], keep=False)]
    # print(duplicate_values[["ADM3_PCODE", "ADM2_PCODE"]].sort_values(by="ADM3_PCODE"))   

    edges = mappings[['ADM2_PCODE', 'ADM3_PCODE']]
    edges = edges.rename(columns={'ADM2_PCODE': 'source', 'ADM3_PCODE': 'target'})
    # edges[["source", "target"]] = edges[["target", "source"]] #Used for producing back-edges
    return edges


def load_data(shape_path_coarse, shape_path_fine, features_path_coarse, features_path_fine, mappings_path, admin_level_coarse, admin_level_fine):

    # Join two edges dataframes which contains edges between admin 2 areas and edges between admin 2 and 3 areas
    edges_coarse = convert_shapefile_to_graph(shape_path_coarse, admin_level_coarse)
    edges_coarse["type"] = "coarse"
    edges_fine = convert_shapefile_to_graph(shape_path_fine, admin_level_fine)
    edges_fine["type"] = "fine"
    edges_mappings = convert_level_mappings_to_edges(mappings_path)
    edges_mappings["type"] = "mappings"
    edges = pd.concat([edges_coarse, edges_mappings])
    # edges = edges_mappings
    
    #Add weights for edges
    weights = {"coarse" : 0.5, "fine" : 0.2, "mappings" : 3}
    # weights = {"coarse" : 1, "mappings" : 1}
    edges["weights_init"] = edges["type"].map(weights)
    edges["weights_init"] = edges["weights_init"] / edges["weights_init"].max()

    #Load node fgeatures for coarse admin and fine admin level and rename to generealize for concat
    col_coarse_pcode = "ADM" + str(admin_level_coarse) + "_PCODE"   
    col_coarse_pt = "ADM" + str(admin_level_coarse) + "_PT"
    
    node_features_coarse = pd.read_csv(features_path_coarse)
    node_features_coarse["admin_level"] = admin_level_coarse
    node_features_coarse = node_features_coarse.rename(columns={col_coarse_pcode : "ADM_PCODE", col_coarse_pt : "ADM_PT"})

    col_fine_pcode = "ADM" + str(admin_level_fine) + "_PCODE"
    col_fine_pt = "ADM" + str(admin_level_fine) + "_PT"
    node_features_fine = pd.read_csv(features_path_fine)
    node_features_fine["admin_level"] = admin_level_fine
    node_features_fine = node_features_fine.rename(columns={col_fine_pcode : "ADM_PCODE", col_fine_pt : "ADM_PT"})

    node_features = pd.concat([node_features_coarse, node_features_fine])
    
    # Check if mismatching districts between shape and feature data
    missing_districts = set(edges["source"]).union(set(edges["target"])) - set(node_features["ADM_PCODE"])
    
    if missing_districts:
        print("Missing districts in the features file: ", missing_districts)

    #Retrieve node features in tensor form
    x_features, y_feature, edges_final, node_features = process_feature_data(node_features, edges)

    x = torch.tensor(x_features, dtype=torch.float)

    #Create edge index
    edge_index = torch.tensor(edges_final[["source_num", "target_num"]].values.T, dtype=torch.long)

    #Edge weights
    edge_weights = torch.tensor(edges_final["weights_init"].values, dtype=torch.float)

    #Produce lap edges
    # lap_edge_index, lap_edge_weight = get_laplacian(edge_index, edge_weights, normalization=None)
    # mask = lap_edge_index[0] != lap_edge_index[1]
    # lap_edge_index = lap_edge_index[:, mask]
    # lap_edge_weight = lap_edge_weight[mask]
    # lap_edge_weight = torch.abs(lap_edge_weight) #Remove negatives
    # edges["lap_weight"] = lap_edge_weight.numpy()
    # edges["weights_init"] = edges["weights_init"] * edges["lap_weight"] #Find new weights
    # edges["weights_init"] = edges["weights_init"] / edges["weights_init"].max() #Normalize weights
    # edge_weights = torch.tensor(edges["weights_init"].values, dtype=torch.float)

    edge_index = torch.tensor(edges_final[["source_num", "target_num"]].values.T, dtype=torch.long)
    
    
    #Produce graph data
    data = Data(x=x, edge_index=edge_index)
    data.edge_weight = edge_weights
    data.y = torch.tensor(y_feature.values, dtype=torch.float)

    admin_level_vals = node_features["admin_level"].values
    data.admin_level = torch.tensor(admin_level_vals, dtype=torch.long)
    data.admin_codes = np.array(node_features.index.values.tolist())
    
    #Train/val/test split
    node_nums = data.num_nodes
    positions = torch.randperm(node_nums)

    
    data.train_mask = (data.admin_level == 2) & (torch.rand(data.num_nodes) < 0.9)
    data.val_mask = (data.admin_level == 2) & ~ data.train_mask
    data.test_mask = (data.admin_level == 3)  # Test on admin level 3 if needed

    #Check number of disconnected nodes
    # connected_nodes = torch.unique(data.edge_index)
    # num_nodes = data.num_nodes
    # disconnected_nodes = torch.tensor([i for i in range(num_nodes) if i not in connected_nodes])
    # print(f"Total nodes: {num_nodes}")
    # print(f"Connected nodes: {connected_nodes.shape[0]}")
    # print(f"Disconnected nodes: {disconnected_nodes.shape[0]}")

    # train_size = int(0.7 * node_nums)
    # val_size = int(0.2 * node_nums)

    # data.train_mask = torch.zeros(node_nums, dtype=torch.bool)
    # data.val_mask = torch.zeros(node_nums, dtype=torch.bool)
    # data.test_mask = torch.zeros(node_nums, dtype=torch.bool)

    # data.train_mask[positions[:train_size]] = True
    # data.val_mask[positions[train_size:train_size + val_size]] = True
    # data.test_mask[positions[train_size + val_size:]] = True

    # admin_mask = (admin_level_vals == 2)  # Mask for admin level 2 nodes
    # data.y[~torch.tensor(admin_mask)] = -1 #Remove all labels for admin level 3 for semi-supervised learning
    return data

def process_feature_data(node_features, edges):
    #Drop nan values and ensure admin level 3 mappings also dropped
    # print("LENGTH", len(node_features.isna().sum()))
    

   # Convert ADM_PCODE to string for consistency
     
    node_features = node_features.copy()
    # node_features = node_features.dropna() 
    node_features["ADM_PCODE"] = node_features["ADM_PCODE"].astype(str)

    nodes_with_nans = node_features[node_features.isna().any(axis=1)]
    removed_pcodes = set(nodes_with_nans["ADM_PCODE"])  

    admin2_nan_pcodes = set(node_features[(node_features["admin_level"] == 2) & (node_features["ADM_PCODE"].isin(removed_pcodes))]["ADM_PCODE"])

    # Efficiently find admin level 3 nodes that belong to these removed admin 2 nodes
    admin3_remove = node_features[
        (node_features["admin_level"] == 3) &
        (node_features["ADM_PCODE"].str.startswith(tuple(admin2_nan_pcodes)))
    ]

    # Add these admin 3 nodes to the removal list
    removed_pcodes.update(admin3_remove["ADM_PCODE"])
    
    node_features = node_features[~node_features["ADM_PCODE"].isin(removed_pcodes)]
    edges = edges[edges["source"].isin(node_features["ADM_PCODE"]) & edges["target"].isin(node_features["ADM_PCODE"])].copy()
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    node_features.set_index("ADM_PCODE", inplace=True)  # Set pcode as index

    node_features["population_density"] = node_features["T_TL"] / node_features["district_area"] #Produce population density
    node_features = node_features[node_features["population_density"] > 0]  # Remove zero-population areas

    y_feature = node_features["population_density"]
    x_unscaled = node_features.drop(columns=["population_density", "ADM_PT", "log_population", "T_TL"])

    #Convert certain variables into densities (This does nothing)
    x_unscaled[['building_count', 'building_area', 'osm_traffic', 'osm_transport', 'osm_places', 'osm_pofw', 'osm_pois', 'osm_railways', 'osm_roads']] = x_unscaled[['building_count', 'building_area', 'osm_traffic', 'osm_transport', 'osm_places', 'osm_pofw', 
                  'osm_pois', 'osm_railways', 'osm_roads']].div(x_unscaled['district_area'], axis=0)
    
    #Drop another unneeded column
    # x_unscaled = x_unscaled.drop(columns=['district_area'])
    ad_levels = x_unscaled["admin_level"].copy()
    x_unscaled = x_unscaled.drop(columns=['admin_level'])
    
    # --------- Visualisation covariates
    #Log population density
    y_feature_final = np.log1p(1 + y_feature)
    original_index = x_unscaled.index  
    # Standardize x (features)
    scaler_vis = MinMaxScaler()
    x_scaled = scaler_vis.fit_transform(x_unscaled)
    # x_scaled = x_unscaled.values
    # Create DataFrame with original index preserved
    x_features_cols_vis = pd.DataFrame(x_scaled, columns=x_unscaled.columns, index=original_index)
    # Add ADM_PCODE and admin_level, ensuring alignment
    x_features_cols_vis['ADM_PCODE'] = original_index  
    x_features_cols_vis["admin_level"] = ad_levels.reindex(original_index)  # Explicitly align admin_level
    # x_features_cols_vis.to_csv("output/csv_files/covariates_normal.csv")

    #Visualise covariates
    # for col in x_features_cols_vis.columns:
    #     visualisations.plot_shape_file_covariates("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", x_features_cols_vis, str(col))
    # ---------

    
    scaler = StandardScaler()
    x_features = scaler.fit_transform(x_unscaled)
    # x_features = x_unscaled.copy()

    #Convert districts in edges to numbers
    area_to_index = {code: idx for idx, code in enumerate(node_features.index)}
    edges = edges[edges["source"].isin(area_to_index) & edges["target"].isin(area_to_index)].copy()

    edges["source_num"] = edges["source"].map(area_to_index)
    edges["target_num"] = edges["target"].map(area_to_index)

    
    return x_features, y_feature_final, edges, node_features


