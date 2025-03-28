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

if __name__ == "__main__":
    set_seed(3) # 3 isn't great, but best so far (At about 109 epochs)
    data = load_data("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", "data/covariates/district/all_features_districts.csv", "data/covariates/postos/all_features_postos.csv", "data/mappings/mozam_admin_2_to_3_mappings.csv", 2, 3)
    num_features = data.x.shape[1]
    
   
    # Define hyperparameter search space
    params = {
        "hidden_size": [1, 2, 4, 8],  # Number of hidden units
        "message_passing_count": [2, 4, 6],  # GNN layers
        "drop_prob": [0.01, 0.1, 0.2, 0.3],  # Dropout rate
        "learning_rate": [1e-1, 1e-2, 5e-2, 1e-3],  # Learning rate
        "weight_decay": [1e-3, 1e-2, 1e-1],  # L2 regularization
    }

    input_size = num_features
    output_size = 1

    hyperparam_combinations = list(itertools.product(*params.values()))
    best_mape = float("inf")
    best_r2 = float("-inf")
    
    best_params = {}

    # ------------ Hyper parameter search
    # for param in hyperparam_combinations:
    #     hidden_size, message_passing_count, drop_prob, learning_rate, weight_decay = param

    #     model_inst = model.GraphSage(input_size, output_size, hidden_size, message_passing_count, drop_prob)

    #     loss_fn = nn.SmoothL1Loss() 
    #     optimizer = torch.optim.Adam(model_inst.parameters(), lr = learning_rate, weight_decay = weight_decay)
    #     loss, val_acc, train_acc, val_r2, train_r2 = train_validate.train_loop(400, model_inst, data, optimizer, loss_fn)

    #     model_inst.eval()
        
    #     if min(val_acc) < best_mape:
    #         index = np.array(val_acc).argmin()
    #         if val_r2[index] > 0.6:
    #             best_mape = min(val_acc)
    #             best_r2 = val_r2[index]
    #             best_params = param
            
    #     print("MAPE: ", min(val_acc), "\nParams: ", param, "\nR2: ", val_r2[np.array(val_acc).argmin()])
    #     print(best_params)
    #     print(best_r2)
    #     print(best_mape)
    # print("Best MAPE: ", best_mape, "\nBest Params: ", best_params, "\nBest R2: ", best_r2)
    # --------------

    print("--------------- START REST OF CODE ---------------")
    
    hidden_size, message_passing_count, drop_prob, learning_rate, weight_decay = 2, 6, 0.01, 5e-2, 1e-2
    # hidden_size = 64
    # message_passing_count = 3
    # drop_prob = 0.1
    # learning_rate = 1e-4
    # weight_decay = 1e-4
    model_inst = model.GraphSage(input_size, output_size, hidden_size, message_passing_count, drop_prob)
    # model_inst = model.LinRegModel(input_size, output_size)

    loss_fn = nn.SmoothL1Loss() 
    # loss_fn = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model_inst.parameters(), lr = learning_rate, weight_decay = weight_decay)
    loss, val_acc, train_acc, val_r2, train_r2 = train_validate.train_loop(200, model_inst, data, optimizer, loss_fn)

    pred, act = train_validate.produce_predictions(data, model_inst, 3)
    visualisations.plot_graph_on_shapefile("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", data, 2, 3)
    visualisations.plot_admin_dists(data)
    visualisations.plot_loss_val_curve(loss, val_acc, train_acc, model_inst)
    visualisations.plot_graph_structure(data)
    visualisations.plot_graph_on_shapefile("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", data, 2, 3)
    visualisations.plot_shape_file("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", 2)
    visualisations.plot_shape_file("data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", 3)
    visualisations.plot_shape_file_predictions("data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", pred, act, 3, data)
    visualisations.plot_residuals(pred, act, model_inst)



