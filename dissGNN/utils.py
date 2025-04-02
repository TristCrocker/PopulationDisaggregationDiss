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

# Function to ensure seed set for reproducibility
def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Produce best score in epochs (Similar to early stop)
def find_best_score(loss, val_mape, train_acc, val_r2, train_r2, val_mae, train_mae):
    best_epoch_info = None
    best_score = float("inf")

    # Loop over epochs
    for epoch in range(len(val_mae)):

        # Find composite score
        score = comp_score(val_mae[epoch], val_mape[epoch], val_r2[epoch])

        # Check if epoch has new best score and R2 above 0
        if score < best_score and val_r2[epoch] > 0 and train_r2[epoch] > 0:
            # Store new scores
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

# Composite score for finding best
def comp_score(mae, mape, r2):
    return mae + mape - (r2 * 100)

# Hyperparameter search function
def hyperparam_search(model_type, input_size, output_size, data, epochs=400):
    # Grid of hyperparameters
    params = {
        "hidden_size": [1, 2, 4, 8],  # Hidden units
        "num_layers": [2, 4, 6],  # GNN layers
        "drop_prob": [0.01, 0.1, 0.2, 0.3],  # Dropout
        "learning_rate": [1e-1, 1e-2, 5e-2, 1e-3],  # Learning rate
        "weight_decay": [1e-3, 1e-2, 1e-1],  # L2 reg
    }

    # Create all combinations
    hyperparam_combinations = list(itertools.product(*params.values()))

    # Init variables
    best_mape = float("inf")
    best_r2 = float("-inf")
    best_score = float("inf")
    best_epoch_info = None
    best_params = {}

    # Loop over combinations
    for param in hyperparam_combinations:
        # Retrieve param
        hidden_size, num_layers, drop_prob, learning_rate, weight_decay = param

        #Create model
        model_inst = model_type(input_size, output_size, hidden_size, num_layers, drop_prob)
        loss_fn = nn.SmoothL1Loss() 
        optimiser = torch.optim.Adam(model_inst.parameters(), lr = learning_rate, weight_decay = weight_decay)

        # Train model on combo
        loss, val_mape, train_acc, val_r2, train_r2, val_mae, train_mae = train_validate.train_loop(epochs, model_inst, data, optimiser, loss_fn)

        model_inst.eval()
        # Loop over epochs
        for epoch in range(len(val_mae)):

            # Find composite score
            score = comp_score(val_mae[epoch], val_mape[epoch], val_r2[epoch])

            #Check if epoch has new best score
            if score < best_score:
                # Update best scores and hyerparams
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