import geopandas as gpd
import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import torch_geometric.utils as ptg_utils
from sklearn.preprocessing import StandardScaler



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


def load_population(args):

    shape_path_coarse = "dataset/population_data/data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp"
    shape_path_fine = "dataset/population_data/data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp"
    features_path_coarse = "dataset/population_data/data/covariates/district/all_features_districts.csv"
    features_path_fine = "dataset/population_data/data/covariates/postos/all_features_postos.csv"
    mappings_path = "dataset/population_data/data/mappings/mozam_admin_2_to_3_mappings.csv"
    admin_level_coarse = 2
    admin_level_fine = 3


    # Join two edges dataframes which contains edges between admin 2 areas and edges between admin 2 and 3 areas
    edges_coarse = convert_shapefile_to_graph(shape_path_coarse, admin_level_coarse)
    edges_coarse["type"] = "coarse"
    edges_fine = convert_shapefile_to_graph(shape_path_fine, admin_level_fine)
    edges_fine["type"] = "fine"
    edges_mappings = convert_level_mappings_to_edges(mappings_path)
    edges_mappings["type"] = "mappings"
    edges = pd.concat([edges_coarse, edges_fine, edges_mappings])

    #Add weights for edges
    weights = {"coarse" : 1.0, "fine" : 0.5, "mappings" : 0.2}
    edges["weights_init"] = edges["type"].map(weights)
    

    #Normalize weights TODO

    #Load node fgeatures for coarse admin and fine admin level and rename to generealize for concat
    col_coarse_pcode = "ADM" + str(admin_level_coarse) + "_PCODE"
    col_coarse_pt = "ADM" + str(admin_level_coarse) + "_PT"
    
    node_features_coarse = pd.read_csv(features_path_coarse)
    node_features_coarse = node_features_coarse.rename(columns={col_coarse_pcode : "ADM_PCODE", col_coarse_pt : "ADM_PT"})
    

    col_fine_pcode = "ADM" + str(admin_level_fine) + "_PCODE"
    col_fine_pt = "ADM" + str(admin_level_fine) + "_PT"
    node_features_fine = pd.read_csv(features_path_fine)
    node_features_fine = node_features_fine.rename(columns={col_fine_pcode : "ADM_PCODE", col_fine_pt : "ADM_PT"})
    

    node_features = pd.concat([node_features_coarse, node_features_fine])
    
    
    # Check if mismatching districts between shape and feature data
    missing_districts = set(edges["source"]).union(set(edges["target"])) - set(node_features["ADM_PCODE"])
    
    if missing_districts:
        print("Missing districts in the features file: ", missing_districts)

    #Retrieve node features in tensor form
    node_features, edges = process_feature_data(node_features, edges)
    x = torch.tensor(node_features.values, dtype=torch.float)

    #Convert districts in edges to numbers
    area_to_index = {code: idx for idx, code in enumerate(node_features.index)}
    edges["source_num"] = edges["source"].map(area_to_index)
    edges["target_num"] = edges["target"].map(area_to_index)
    print(edges)

    #Create edge index
    edge_index = torch.tensor(edges[["source_num", "target_num"]].values.T, dtype=torch.long)

    # Ensure edges are undirected and remove self-loops
    edge_index = ptg_utils.to_undirected(edge_index)
    edge_index = ptg_utils.remove_self_loops(edge_index)[0]

    #Edge weights
    edge_weights = torch.tensor(edges["weights_init"].values, dtype=torch.float)

    #Produce graph data
    data = Data(x=x, edge_index=edge_index)
    data.edge_weight = edge_weights
    data.y = torch.tensor(node_features["population_density"].values, dtype=torch.float)
    data.y = torch.log1p(data.y)

    #Standardize data
    data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-6)

    #Train/val/test split
    node_nums = data.num_nodes
    positions = torch.randperm(node_nums)

    train_size = int(0.7 * node_nums)
    val_size = int(0.2 * node_nums)

    data.train_mask = torch.zeros(node_nums, dtype=torch.bool)
    data.val_mask = torch.zeros(node_nums, dtype=torch.bool)
    data.test_mask = torch.zeros(node_nums, dtype=torch.bool)

    data.train_mask[positions[:train_size]] = True
    data.val_mask[positions[train_size:train_size + val_size]] = True
    data.test_mask[positions[train_size + val_size:]] = True

    data.num_nodes = data.x.shape[0]  # Ensure num_nodes is explicitly set

    final_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=data.y, num_nodes = data.x.shape[0])  # Ensure num_nodes is explicitly set)

    return final_data

def process_feature_data(node_features, edges):
    node_features = node_features.dropna()  # Drop NaN values

    node_features["population_density"] = node_features["T_TL"] / node_features["district_area"] #Produce population density
    node_features = node_features[node_features["population_density"] > 0]  # Remove zero-population areas
    edges = edges[edges["source"].isin(node_features["ADM_PCODE"]) & edges["target"].isin(node_features["ADM_PCODE"])] #Remove edges no longer valid

    node_features.set_index("ADM_PCODE", inplace=True) #Set pcode as index
    node_features = node_features.drop(columns=["ADM_PT", "log_population", "T_TL"]) #Remove unwated features

    scaler = StandardScaler()
    node_features.iloc[:, :] = scaler.fit_transform(node_features)
    
    return node_features, edges


