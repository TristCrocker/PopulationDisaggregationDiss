import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

# Train function for a single epoch
def train(model, optimiser, data, loss_fn):
    # Setup
    optimiser.zero_grad()

    # Produce predictions
    output = model(data.x, data.edge_index, data.edge_weight)

    # Retrieve predictions and actual with train set
    train_preds = output[data.train_mask].squeeze()
    train_labels = data.y[data.train_mask]

    # Retrieve epoch loss
    loss = loss_fn(train_preds, train_labels)
    loss.backward()
    optimiser.step()

    return loss.item()

# Validation function
def validate(model, data, mask, e=1e-6):
    # Produce predictions
    output = model(data.x, data.edge_index, data.edge_weight)

    # Compute MAPE
    actual = torch.expm1(data.y[mask])  # Convert log values back to original scale
    predicted = torch.expm1(output[mask].squeeze()) # Convert log predictions back to original scale

    # Compute MAE
    mae = torch.abs(predicted - actual).mean().item()

    # Compute MAPE
    valid_mask = actual > e
    mape = (torch.abs((predicted[valid_mask] - actual[valid_mask]) / actual[valid_mask])).mean().item() * 100

    # Compute R2
    ss_total = torch.sum((actual - actual.mean()) ** 2)
    ss_residual = torch.sum((actual - predicted) ** 2)
    r2 = 1 - (ss_residual / (ss_total)).item() 

    return mae, mape, r2 

# Training loop
def train_loop(num_epochs, model, data, optimiser, loss_fn):
    # Init variables to store metrics
    loss_arr = []
    val_acc_arr = []
    train_acc_arr = []
    train_r2_arr = []
    val_r2_arr = []
    train_mae_arr = []
    val_mae_arr = []
    
    for param in model.parameters():
        param.requires_grad = True
    
    # Epoch loop
    for epoch in range(num_epochs):
        model.train()

        # Train model
        loss = train(model, optimiser, data, loss_fn)

        # Store loss
        loss_arr.append(loss)

        # Validate on train (admin level 2)
        mae, mape, r2 = validate(model, data, data.train_mask)
        print("Train - Epoch Number: ", epoch, ", Loss: ", loss, ", MAE: ", mae, ", MAPE: ", mape, ", R^2: ", r2, ".")
        train_acc_arr.append(mape)
        train_r2_arr.append(r2)
        train_mae_arr.append(mae)
        
        # Validate model on test set (admin level 3)
        model.eval()
        with torch.no_grad():
            mae, mape, r2 = validate(model, data, data.test_mask)
            print("Test - Epoch Number: ", epoch, ", Loss: ", loss, ", MAE: ", mae, ", MAPE: ", mape, ", R^2: ", r2, ".")
            val_acc_arr.append(mape)
            val_r2_arr.append(r2)
            val_mae_arr.append(mae)

    return loss_arr, val_acc_arr, train_acc_arr, val_r2_arr, train_r2_arr, val_mae_arr, train_mae_arr

# Produce test predictions (Admin level 3)
def produce_predictions(data, model, admin_level):
    model.eval()
    with torch.no_grad():
        # Produce predictions
        predictions = model(data.x, data.edge_index, data.edge_weight)

    predictions_final = torch.expm1(predictions[data.test_mask]).cpu().numpy().flatten()
    actual_final = torch.expm1(data.y[data.test_mask]).cpu().numpy().flatten()

    return predictions_final, actual_final  