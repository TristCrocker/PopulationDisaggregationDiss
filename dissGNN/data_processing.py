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

def load_data(shape_path, features_path, admin_level):
    edges = convert_shapefile_to_graph(shape_path, admin_level)
    node_features = pd.read_csv(features_path)
    node_features = process_feature_data(node_features
                                         )
    # Check if mismatching districts between shape and feature data
    missing_districts = set(edges["source"]).union(set(edges["target"])) - set(node_features["ADM2_PCODE"])
    if missing_districts:
        print("Missing districts in the features file: ", missing_districts)

    x = torch.tensor(node_features, dtype=torch.float)

    #Convert districts in edges to numbers
    district_to_index = {code: idx for idx, code in enumerate(node_features.index)}
    edges["source_idx"] = edges["source"].map(district_to_index)
    edges["target_idx"] = edges["target"].map(district_to_index)

    #Create edge index
    edge_index = torch.tensor(edges[["source_idx", "target_idx"]].values.T, dtype=torch.long)

    #Produce graph data
    data = Data(x=x, edge_index=edge_index)
    data.y = torch.tensor(node_features["T_TL"].values, dtype=torch.float)

    return data

def process_feature_data(node_features):
      node_features = node_features.drop(columns=["ADM2_PT", "ADM2_PCODE", "log_population"]).values
      return node_features


data = load_data("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/covariates/district/all_features_districts.csv", 2)
