import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

def train(model, optimizer, data, loss_fn):
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_weight)
    train_preds = output[data.train_mask].squeeze()
    train_labels = data.y[data.train_mask]

    # valid_mask = train_labels != -1  # Only keep admin level 2 nodes

    # loss = loss_fn(torch.expm1(train_preds[valid_mask]), torch.expm1(train_labels[valid_mask]))
    loss = loss_fn(train_preds, train_labels)
    # loss = loss_fn(train_preds, train_labels)
    
    loss.backward()

    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Gradient Norm [{name}]: {torch.norm(param.grad).item()}")

    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.4)
    optimizer.step()
    return loss.item()

def validate(model, data, mask, e=1e-6):
    output = model(data.x, data.edge_index, data.edge_weight)

    # Compute MAPE
    actual = torch.expm1(data.y[mask])  # Convert log values back to original scale
    predicted = torch.expm1(output[mask].squeeze()) # Convert log predictions back to original scale

    # actual = data.y[mask]  # Convert log values back to original scale
    # predicted = output[mask].squeeze()  # Convert log predictions back to original scale

    # Compute MAE
    mae = torch.abs(predicted - actual).mean().item()

    # Compute RMSE
    mse = torch.mean((predicted - actual) ** 2)
    rmse = torch.sqrt(mse).item()

    valid_mask = actual > e
    mape = (torch.abs((predicted[valid_mask] - actual[valid_mask]) / actual[valid_mask])).mean().item() * 100

    # Compute R² (Coefficient of Determination)
    ss_total = torch.sum((actual - actual.mean()) ** 2)
    ss_residual = torch.sum((actual - predicted) ** 2)
    r2 = 1 - (ss_residual / (ss_total)).item()  # Add epsilon to avoid div by zero

    return mae, mape, r2 

def train_loop(num_epochs, model, data, optimizer, loss_fn):
    loss_arr = []
    val_acc_arr = []
    train_acc_arr = []
    train_r2_arr = []
    val_r2_arr = []
    
    for param in model.parameters():
        param.requires_grad = True  # Force enable gradients
    
    for epoch in range(num_epochs):
        model.train()
        loss = train(model, optimizer, data, loss_fn)
        loss_arr.append(loss)
        mae, mape, r2 = validate(model, data, data.train_mask)
        print("Train - Epoch Number: ", epoch, ", Loss: ", loss, ", MAE: ", mae, ", MAPE: ", mape, ", R^2: ", r2, ".")
        train_acc_arr.append(mape)
        train_r2_arr.append(r2)
        

        model.eval()
        with torch.no_grad():
           
            mae, mape, r2 = validate(model, data, data.test_mask)
            print("Test - Epoch Number: ", epoch, ", Loss: ", loss, ", MAE: ", mae, ", MAPE: ", mape, ", R^2: ", r2, ".")
            val_acc_arr.append(mape)
            val_r2_arr.append(r2)

    return loss_arr, val_acc_arr, train_acc_arr, val_r2_arr, train_r2_arr


def produce_predictions(data, model, admin_level):
    model.eval()
    
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, data.edge_weight)


    predictions_final = torch.expm1(predictions[data.test_mask]).cpu().numpy().flatten()
    actual_final = torch.expm1(data.y[data.test_mask]).cpu().numpy().flatten()

    print("Raw Predictions (before expm1):", predictions_final )
    print("Actual Values (before expm1):", actual_final)


    # predictions_final = predictions[data.test_mask].cpu().numpy().flatten()
    # actual_final = data.y[data.test_mask].cpu().numpy().flatten()

    return predictions_final, actual_final  