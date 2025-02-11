import torch.nn as nn
import model
import torch
import train_validate
from data_processing import load_data
import itertools

if __name__ == "__main__":
    data = load_data("data/shapefiles/admin_2/moz_admbnda_adm2_ine_20190607.shp", "data/shapefiles/admin_3/moz_admbnda_adm3_ine_20190607.shp", "data/covariates/district/all_features_districts.csv", "data/covariates/postos/all_features_postos.csv", "data/mappings/mozam_admin_2_to_3_mappings.csv", 2, 3)
    num_features = data.x.shape[1]
   
    # Define hyperparameter search space
    params = {
        "hidden_size": [16, 32, 64, 128],  # Number of hidden units
        "message_passing_count": [3, 6, 9],  # GNN layers
        "drop_prob": [0.01, 0.1, 0.2, 0.3],  # Dropout rate
        "learning_rate": [1e-4, 5e-5, 1e-5, 1e-6],  # Learning rate
        "weight_decay": [1e-4, 1e-3, 1e-2],  # L2 regularization
    }

    input_size = num_features
    output_size = 1

    hyperparam_combinations = list(itertools.product(*params.values()))
    best_mape = float("inf")
    best_params = {}

    for param in hyperparam_combinations:
        hidden_size, message_passing_count, drop_prob, learning_rate, weight_decay = param

        model_inst = model.GCN(input_size, output_size, hidden_size, message_passing_count, drop_prob)

        loss_fn = nn.SmoothL1Loss() 
        optimizer = torch.optim.Adam(model_inst.parameters(), lr = learning_rate, weight_decay = weight_decay)
        mae, mape, rmse, r2 = train_validate.train_loop(400, model_inst, data, optimizer, loss_fn)

        model_inst.eval()

        if mape < best_mape:
            best_mape = mape
            best_params = params
            
        print(best_params)
        print(best_mape)
    
    print(best_mape, "\n", best_params)



