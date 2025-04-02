import torch.nn as nn
import model
import torch
import train_validate
from data_processing import load_data
import itertools
import visualisations
import pandas as pd
import numpy as np
import random
from utils import *

# Main function
if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(3) 

    #Load data
    data = load_data("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", "data/covariates/district/all_features_districts.csv", "data/covariates/postos/all_features_postos.csv", "data/mappings/mozam_admin_2_to_3_mappings.csv", 2, 3)
    num_features = data.x.shape[1]
    input_size = num_features
    output_size = 1

    # ---------------- Hyperparam search -------------
    # TransformerData, trans_params = hyperparam_search(model.TransformerNet, input_size, output_size, data, epochs=400)
    # GCN2ConvData, gcn2_params = hyperparam_search(model.GCN2Net, input_size, output_size, data, epochs=400)
    # GAT2Data, gat2_params = hyperparam_search(model.GATv2Net, input_size, output_size, data, epochs=400)
    # SageData, sage_params = hyperparam_search(model.GraphSage, input_size, output_size, data, epochs=400)
    # print("Trans: ", TransformerData, trans_params)
    # print("GCN2: ", GCN2ConvData, gcn2_params)
    # print("GAT2: ", GAT2Data, gat2_params)
    # print("Sage: ", SageData, sage_params)
    # ---------------- Hyperparam search -------------


    #Setup all models with best params from hyperparam search
    hidden_size, num_layers, drop_prob, learning_rate, weight_decay = 8, 2, 0.1, 0.1, 0.01
    model_inst_TRAN = model.TransformerNet(input_size, output_size, hidden_size, num_layers, drop_prob)
    optimizer_TRAN = torch.optim.Adam(model_inst_TRAN.parameters(), lr = learning_rate, weight_decay = weight_decay)

    hidden_size, num_layers, drop_prob, learning_rate, weight_decay = 8, 2, 0.1, 0.1, 0.01
    model_inst_GCN2 = model.GCN2Net(input_size, output_size, hidden_size, num_layers, drop_prob)
    optimizer_GCN2 = torch.optim.Adam(model_inst_GCN2.parameters(), lr = learning_rate, weight_decay = weight_decay)

    hidden_size, num_layers, drop_prob, learning_rate, weight_decay = 2, 2, 0.01, 0.1, 0.01
    model_inst_GAT2 = model.GATv2Net(input_size, output_size, hidden_size, num_layers, drop_prob)
    optimizer_GAT2 = torch.optim.Adam(model_inst_GAT2.parameters(), lr = learning_rate, weight_decay = weight_decay)

    hidden_size, num_layers, drop_prob, learning_rate, weight_decay = 4, 6, 0.01, 0.01, 0.001
    model_inst_SAGE = model.GraphSage(input_size, output_size, hidden_size, num_layers, drop_prob)
    optimizer_SAGE = torch.optim.Adam(model_inst_SAGE.parameters(), lr = learning_rate, weight_decay = weight_decay)
    # model_inst_lin = model.LinRegModel(input_size, output_size) # Linear reg model

    # Set huber loss
    loss_fn = nn.SmoothL1Loss() 
    
    #Train all models and find best metric scores (These are not the optimal results)
    loss_TRAN, val_acc_TRAN, train_acc_TRAN, val_r2_TRAN, train_r2_TRAN, val_mae_TRAN, train_mae_TRAN = train_validate.train_loop(400, model_inst_TRAN, data, optimizer_TRAN, loss_fn)
    loss_GCN2, val_acc_GCN2, train_acc_GCN2, val_r2_GCN2, train_r2_GCN2, val_mae_GCN2, train_mae_GCN2 = train_validate.train_loop(400, model_inst_GCN2, data, optimizer_GCN2, loss_fn)
    loss_GAT2, val_acc_GAT2, train_acc_GAT2, val_r2_GAT2, train_r2_GAT2, val_mae_GAT2, train_mae_GAT2 = train_validate.train_loop(400, model_inst_GAT2, data, optimizer_GAT2, loss_fn)
    loss_SAGE, val_acc_SAGE, train_acc_SAGE, val_r2_SAGE, train_r2_SAGE, val_mae_SAGE, train_mae_SAGE = train_validate.train_loop(400, model_inst_SAGE, data, optimizer_SAGE, loss_fn)
    
    
    print("\nTransformer Metrics: \n", find_best_score(loss_TRAN, val_acc_TRAN, train_acc_TRAN, val_r2_TRAN, train_r2_TRAN, val_mae_TRAN, train_mae_TRAN))
    print("\nGCNv2 Metrics: \n", find_best_score(loss_GCN2, val_acc_GCN2, train_acc_GCN2, val_r2_GCN2, train_r2_GCN2, val_mae_GCN2, train_mae_GCN2))
    print("\nGATv2 Metrics: \n", find_best_score(loss_GAT2, val_acc_GAT2, train_acc_GAT2, val_r2_GAT2, train_r2_GAT2, val_mae_GAT2, train_mae_GAT2))
    print("\nGraphSage Metrics: \n", find_best_score(loss_SAGE, val_acc_SAGE, train_acc_SAGE, val_r2_SAGE, train_r2_SAGE, val_mae_SAGE, train_mae_SAGE))

    # Produce predictions for all models
    pred_TRAN, act_TRAN = train_validate.produce_predictions(data, model_inst_TRAN, 3)
    pred_GCN2, act_GCN2 = train_validate.produce_predictions(data, model_inst_GCN2, 3)
    pred_GAT2, act_GAT2 = train_validate.produce_predictions(data, model_inst_GAT2, 3)
    pred_SAGE, act_SAGE = train_validate.produce_predictions(data, model_inst_SAGE, 3)

    # ---------------- Visualisations --------------
    # visualisations.plot_graph_on_shapefile("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", data, 2, 3)
    # visualisations.plot_admin_dists(data)
    # visualisations.plot_graph_structure(data)

    # # Predictions on shapefile
    # visualisations.plot_shape_file_predictions("data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", pred_TRAN, act_TRAN, 3, data, model_inst_TRAN)
    # visualisations.plot_shape_file_predictions("data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", pred_GCN2, act_GCN2, 3, data, model_inst_GCN2)
    # visualisations.plot_shape_file_predictions("data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", pred_GAT2, act_GAT2, 3, data, model_inst_GAT2)
    # visualisations.plot_shape_file_predictions("data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", pred_SAGE, act_SAGE, 3, data, model_inst_SAGE)

    # #Residuals
    # visualisations.plot_residuals(pred_TRAN, act_TRAN, model_inst_TRAN)
    # visualisations.plot_residuals(pred_GCN2, act_GCN2, model_inst_GCN2)
    # visualisations.plot_residuals(pred_GAT2, act_GAT2, model_inst_GAT2)
    # visualisations.plot_residuals(pred_SAGE, act_SAGE, model_inst_SAGE)

    # visualisations.plot_acc_loss_over_epochs([loss_TRAN, loss_GCN2, loss_GAT2, loss_SAGE], [model_inst_TRAN, model_inst_GCN2, model_inst_GAT2, model_inst_SAGE])
    # ---------------- Visualisations --------------




