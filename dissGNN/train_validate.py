import torch
from sklearn.preprocessing import StandardScaler

def train(model, optimizer, data, loss_fn):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_weight)
    loss = loss_fn(torch.expm1(output[data.train_mask].squeeze()), torch.expm1(data.y[data.train_mask]))
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data, e=1e-6):

    with torch.no_grad():

        output = model(data.x, data.edge_index, data.edge_weight).squeeze()


        # Compute MAPE
        actual = torch.expm1(data.y[data.val_mask])  # Convert log values back to original scale
        # actual = scaler.inverse_transform(actual)
        predicted = torch.expm1(output[data.val_mask])  # Convert log predictions back to original scale
        # predicted = scaler.inverse_transform(predicted)

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
    for epoch in range(num_epochs):
        loss = train(model, optimizer, data, loss_fn)
        mae, mape, rmse, r2 = validate(model, data)
        
        print("Epoch Number: ", epoch, ", Loss: ", loss, ", Validation MAE: ", mae, ", Validation MAPE: ", mape, ", Validation RMSE: ", rmse, ", Validation R^2: ", r2, ".")
    
    return mae, mape, rmse, r2


    