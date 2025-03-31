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



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_best_score(loss, val_mape, train_acc, val_r2, train_r2, val_mae, train_mae):
    best_epoch_info = None
    best_score = float("inf")
    for epoch in range(len(val_mae)):
        score = comp_score(val_mae[epoch], val_mape[epoch], val_r2[epoch])
        if score < best_score and val_r2[epoch] > 0 and train_r2[epoch] > 0:
            best_score = score
            best_epoch_info = {
                "epoch": epoch,
                "train_mae": train_mae[epoch],
                "val_mae": val_mae[epoch],
                "train_mape": train_acc[epoch],
                "val_mape": val_mape[epoch],
                "train_r2": train_r2[epoch],
                "val_r2": val_r2[epoch]
            }

    return best_epoch_info


def comp_score(mae, mape, r2):
    return mae + mape - (r2 * 100)

def hyperparam_search(model_type, input_size, output_size, epochs=400):
    params = {
        "hidden_size": [1, 2, 4, 8],  # Number of hidden units
        "message_passing_count": [2, 4, 6],  # GNN layers
        "drop_prob": [0.01, 0.1, 0.2, 0.3],  # Dropout rate
        "learning_rate": [1e-1, 1e-2, 5e-2, 1e-3],  # Learning rate
        "weight_decay": [1e-3, 1e-2, 1e-1],  # L2 regularization
    }



    hyperparam_combinations = list(itertools.product(*params.values()))
    best_mape = float("inf")
    best_r2 = float("-inf")
    best_score = float("inf")
    best_epoch_info = None
    
    best_params = {}


    # ------------ Hyper parameter search --------------
    for param in hyperparam_combinations:
        hidden_size, message_passing_count, drop_prob, learning_rate, weight_decay = param

        model_inst = model_type(input_size, output_size, hidden_size, message_passing_count, drop_prob)

        loss_fn = nn.SmoothL1Loss() 
        optimizer = torch.optim.Adam(model_inst.parameters(), lr = learning_rate, weight_decay = weight_decay)
        loss, val_mape, train_acc, val_r2, train_r2, val_mae, train_mae = train_validate.train_loop(epochs, model_inst, data, optimizer, loss_fn)

        model_inst.eval()
        
        for epoch in range(len(val_mae)):
            score = comp_score(val_mae[epoch], val_mape[epoch], val_r2[epoch])
            if score < best_score:
                best_score = score
                best_epoch_info = {
                    "epoch": epoch,
                    "val_mae": val_mae[epoch],
                    "val_mape": val_mape[epoch],
                    "val_r2": val_r2[epoch],
                    "train_mae": train_mae[epoch],
                    "train_mape": train_acc[epoch],
                    "train_r2": train_r2[epoch],
                }
                best_params = param
    return best_epoch_info, best_params

if __name__ == "__main__":
    set_seed(3) # 3 isn't great, but best so far (At about 109 epochs)
    data = load_data("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", "data/covariates/district/all_features_districts.csv", "data/covariates/postos/all_features_postos.csv", "data/mappings/mozam_admin_2_to_3_mappings.csv", 2, 3)
    num_features = data.x.shape[1]

    input_size = num_features
    output_size = 1

    hidden_size, message_passing_count, drop_prob, learning_rate, weight_decay = 8, 2, 0.1, 0.1, 0.01

    # ---------------- Hyperparam search -------------
    # TransformerData, trans_params = hyperparam_search(model.TransformerNet, input_size, output_size, epochs=400)
    # GCN2ConvData, gcn2_params = hyperparam_search(model.GCN2Net, input_size, output_size, epochs=400)
    # GAT2Data, gat2_params = hyperparam_search(model.GATv2Net, input_size, output_size, epochs=400)
    # SageData, sage_params = hyperparam_search(model.GraphSage, input_size, output_size, epochs=400)
    # print("Trans: ", TransformerData, trans_params)
    # print("GCN2: ", GCN2ConvData, gcn2_params)
    # print("GAT2: ", GAT2Data, gat2_params)
    # print("Sage: ", SageData, sage_params)


    #Setup all models
    hidden_size, message_passing_count, drop_prob, learning_rate, weight_decay = 8, 2, 0.1, 0.1, 0.01
    model_inst_TRAN = model.TransformerNet(input_size, output_size, hidden_size, message_passing_count, drop_prob)
    optimizer_TRAN = torch.optim.Adam(model_inst_TRAN.parameters(), lr = learning_rate, weight_decay = weight_decay)

    hidden_size, message_passing_count, drop_prob, learning_rate, weight_decay = 8, 2, 0.1, 0.1, 0.01
    model_inst_GCN2 = model.GCN2Net(input_size, output_size, hidden_size, message_passing_count, drop_prob)
    optimizer_GCN2 = torch.optim.Adam(model_inst_GCN2.parameters(), lr = learning_rate, weight_decay = weight_decay)

    hidden_size, message_passing_count, drop_prob, learning_rate, weight_decay = 2, 2, 0.01, 0.1, 0.01
    model_inst_GAT2 = model.GATv2Net(input_size, output_size, hidden_size, message_passing_count, drop_prob)
    optimizer_GAT2 = torch.optim.Adam(model_inst_GAT2.parameters(), lr = learning_rate, weight_decay = weight_decay)

    hidden_size, message_passing_count, drop_prob, learning_rate, weight_decay = 4, 6, 0.01, 0.01, 0.001
    model_inst_SAGE = model.GraphSage(input_size, output_size, hidden_size, message_passing_count, drop_prob)
    optimizer_SAGE = torch.optim.Adam(model_inst_SAGE.parameters(), lr = learning_rate, weight_decay = weight_decay)
    # model_inst_lin = model.LinRegModel(input_size, output_size)

    loss_fn = nn.SmoothL1Loss() 
    
    #Train all models
    loss_TRAN, val_acc, train_acc, val_r2, train_r2, val_mae, train_mae = train_validate.train_loop(150, model_inst_TRAN, data, optimizer_TRAN, loss_fn)
    loss_GCN2, val_acc, train_acc, val_r2, train_r2, val_mae, train_mae = train_validate.train_loop(150, model_inst_GCN2, data, optimizer_GCN2, loss_fn)
    loss_GAT2, val_acc, train_acc, val_r2, train_r2, val_mae, train_mae = train_validate.train_loop(150, model_inst_GAT2, data, optimizer_GAT2, loss_fn)
    loss_SAGE, val_acc, train_acc, val_r2, train_r2, val_mae, train_mae = train_validate.train_loop(150, model_inst_SAGE, data, optimizer_SAGE, loss_fn)
    # best_epoch_score = find_best_score(loss, val_acc, train_acc, val_r2, train_r2, val_mae, train_mae)

    #Produce predictions for all models
    pred_TRAN, act_TRAN = train_validate.produce_predictions(data, model_inst_TRAN, 3)
    pred_GCN2, act_GCN2 = train_validate.produce_predictions(data, model_inst_GCN2, 3)
    pred_GAT2, act_GAT2 = train_validate.produce_predictions(data, model_inst_GAT2, 3)
    pred_SAGE, act_SAGE = train_validate.produce_predictions(data, model_inst_SAGE, 3)

    # Visualisations
    visualisations.plot_graph_on_shapefile("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", data, 2, 3)
    visualisations.plot_admin_dists(data)
    visualisations.plot_graph_structure(data)

    # Predictions on shapefile
    visualisations.plot_shape_file_predictions("data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", pred_TRAN, act_TRAN, 3, data, model_inst_TRAN)
    visualisations.plot_shape_file_predictions("data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", pred_GCN2, act_GCN2, 3, data, model_inst_GCN2)
    visualisations.plot_shape_file_predictions("data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", pred_GAT2, act_GAT2, 3, data, model_inst_GAT2)
    visualisations.plot_shape_file_predictions("data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", pred_SAGE, act_SAGE, 3, data, model_inst_SAGE)

    #Residuals
    visualisations.plot_residuals(pred_TRAN, act_TRAN, model_inst_TRAN)
    visualisations.plot_residuals(pred_GCN2, act_GCN2, model_inst_GCN2)
    visualisations.plot_residuals(pred_GAT2, act_GAT2, model_inst_GAT2)
    visualisations.plot_residuals(pred_SAGE, act_SAGE, model_inst_SAGE)

    visualisations.plot_acc_loss_over_epochs([loss_TRAN, loss_GCN2, loss_GAT2, loss_SAGE], [model_inst_TRAN, model_inst_GCN2, model_inst_GAT2, model_inst_SAGE])




