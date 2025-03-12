import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

def train(model, optimizer, data, loss_fn):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)

    train_preds = output[data.train_mask].squeeze()
    train_labels = data.y[data.train_mask]

    # valid_mask = train_labels != -1  # Only keep admin level 2 nodes

    # loss = loss_fn(torch.expm1(train_preds[valid_mask]), torch.expm1(train_labels[valid_mask]))
    loss = loss_fn(torch.expm1(train_preds), torch.expm1(train_labels))

    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
    optimizer.step()
    return loss.item()

def validate(model, data, mask, e=1e-6):
    model.eval()

    output = model(data.x, data.edge_index).squeeze()

    # Compute MAPE
    actual = torch.expm1(data.y[mask])  # Convert log values back to original scale
    predicted = torch.expm1(output[mask])  # Convert log predictions back to original scale

    # actual = data.y[mask]  # Convert log values back to original scale
    # predicted = output[mask]  # Convert log predictions back to original scale

    # Compute MAE
    mae = torch.abs(predicted - actual).mean().item()

    # Compute RMSE
    mse = torch.mean((predicted - actual) ** 2)
    rmse = torch.sqrt(mse).item()

    # Compute MAPE (Avoid division by zero)
    mape = (torch.abs((predicted - actual) / (actual + e))).mean().item() * 100  

    # Compute R² (Coefficient of Determination)
    ss_total = torch.sum((actual - actual.mean()) ** 2)
    ss_residual = torch.sum((actual - predicted) ** 2)
    r2 = 1 - (ss_residual / (ss_total + e)).item()  # Add epsilon to avoid div by zero

    return mae, mape, rmse, r2  

def train_loop(num_epochs, model, data, optimizer, loss_fn):
    loss_arr = []
    val_acc_arr = []
    train_acc_arr = []
    for epoch in range(num_epochs):
        loss = train(model, optimizer, data, loss_fn)
        loss_arr.append(loss)
        mae, mape, rmse, r2 = validate(model, data, data.val_mask)
        print("Val - Epoch Number: ", epoch, ", Loss: ", loss, ", MAE: ", mae, ", MAPE: ", mape, ", RMSE: ", rmse, ", R^2: ", r2, ".")
        val_acc_arr.append(mape)
        mae, mape, rmse, r2 = validate(model, data, data.train_mask)
        print("Train - Epoch Number: ", epoch, ", Loss: ", loss, ", MAE: ", mae, ", MAPE: ", mape, ", RMSE: ", rmse, ", R^2: ", r2, ".")
        train_acc_arr.append(mape)
    
    return loss_arr, val_acc_arr, train_acc_arr


def produce_predictions(data, model, admin_level):
    model.eval()
    mask_admin_level = (data.admin_level == admin_level)
    
    with torch.no_grad():
        predictions = model(data.x, data.edge_index)

    predictions_final = torch.expm1(predictions[mask_admin_level]).cpu().numpy().flatten()
    actual_final = torch.expm1(data.y[mask_admin_level]).cpu().numpy().flatten()

    return predictions_final, actual_final  