import geopandas as gpd
import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data


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

    return pd.DataFrame(edges, columns=["source", "target"])

def convert_level_mappings_to_edges(mappings_path):
    
    mappings = pd.read_csv(mappings_path)
    edges = mappings[['ADM2_PCODE', 'ADM3_PCODE']]
    edges = edges.rename(columns={'ADM2_PCODE': 'source', 'ADM3_PCODE': 'target'})

    return edges


def load_data(shape_path_coarse, shape_path_fine, features_path_coarse, features_path_fine, mappings_path, admin_level_coarse, admin_level_fine):

    # Join two edges dataframes which contains edges between admin 2 areas and edges between admin 2 and 3 areas
    edges_coarse = convert_shapefile_to_graph(shape_path_coarse, admin_level_coarse)
    edges_fine = convert_shapefile_to_graph(shape_path_fine, admin_level_fine)
    edges_mappings = convert_level_mappings_to_edges(mappings_path)
    edges = pd.concat([edges_coarse, edges_fine, edges_mappings])

    #Load node fgeatures for coarse admin and fine admin level and rename to generealize for concat
    col_coarse_pcode = "ADM" + str(admin_level_coarse) + "_PCODE"
    col_coarse_pt = "ADM" + str(admin_level_coarse) + "_PT"
    
    node_features_coarse = pd.read_csv(features_path_coarse)
    node_features_coarse = node_features_coarse.rename(columns={col_coarse_pcode : "ADM_PCODE", col_coarse_pt : "ADM_PT"})
    print(node_features_coarse)

    col_fine_pcode = "ADM" + str(admin_level_fine) + "_PCODE"
    col_fine_pt = "ADM" + str(admin_level_fine) + "_PT"
    node_features_fine = pd.read_csv(features_path_fine)
    node_features_fine = node_features_fine.rename(columns={col_fine_pcode : "ADM_PCODE", col_fine_pt : "ADM_PT"})
    print(node_features_fine)

    node_features = pd.concat([node_features_coarse, node_features_fine])
    print(node_features)
    
    # Check if mismatching districts between shape and feature data
    missing_districts = set(edges["source"]).union(set(edges["target"])) - set(node_features["ADM_PCODE"])
    
    if missing_districts:
        print("Missing districts in the features file: ", missing_districts)

    node_features = process_feature_data(node_features, admin_level_coarse, admin_level_fine)
    x = torch.tensor(node_features.values, dtype=torch.float)

    #Convert districts in edges to numbers
    district_to_index = {code: idx for idx, code in enumerate(node_features.index)}
    edges["source_idx"] = edges["source"].map(district_to_index)
    edges["target_idx"] = edges["target"].map(district_to_index)

    #Create edge index
    edge_index = torch.tensor(edges[["source_idx", "target_idx"]].values.T, dtype=torch.long)

    #Produce graph data
    data = Data(x=x, edge_index=edge_index)
    data.y = torch.tensor(node_features["T_TL"].values, dtype=torch.float)

    #Standardize data
    data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-6)

    #Train/val/test split
    node_nums = data.num_nodes
    positions = torch.randperm(node_nums)

    train_size = int(0.7 * node_nums)
    val_size = int(0.2 * node_nums)
    test_size = node_nums - train_size - val_size

    data.train_mask = torch.zeros(node_nums, dtype=torch.bool)
    data.val_mask = torch.zeros(node_nums, dtype=torch.bool)
    data.test_mask = torch.zeros(node_nums, dtype=torch.bool)

    data.train_mask[positions[:train_size]] = True
    data.val_mask[positions[train_size:train_size + val_size]] = True
    data.test_mask[positions[train_size + val_size:]] = True

    return data

def process_feature_data(node_features):


    node_features = node_features.drop(columns=["ADM_PCODE", "ADM_PT", "log_population"])
    node_features = node_features.fillna(0)  # Replace NaN values with 0

    return node_features


