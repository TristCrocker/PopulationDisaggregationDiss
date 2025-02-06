import torch

def train(model, optimizer, data, loss_fn):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_weight)
    loss = loss_fn(torch.expm1(output[data.train_mask].squeeze()), torch.expm1(data.y[data.train_mask]))
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data, e=1e-6):
    model.eval()

    with torch.no_grad():

        output = model(data.x, data.edge_index, data.edge_weight).squeeze()

        # Compute MAPE
        actual = torch.expm1(data.y[data.val_mask])  # Convert log values back to original scale
        predicted = torch.expm1(output[data.val_mask])  # Convert log predictions back to original scale
        

        # Debug: Print max and min values
        print(f"Max Actual: {actual.max().item()}, Min Actual: {actual.min().item()}")
        print(f"Max Predicted: {predicted.max().item()}, Min Predicted: {predicted.min().item()}")
        
        # Check if any actual values are zero
        zero_count = (actual == 0).sum().item()
        print(f"Zero Actual Values: {zero_count}")


        mape = (torch.abs((predicted - actual) / actual + e)).mean().item() * 100  # Convert to percentage

    return mape

def train_loop(num_epochs, model, data, optimizer, loss_fn):
    for epoch in range(num_epochs):
        loss = train(model, optimizer, data, loss_fn)
        accuracy = validate(model, data)
        print("Epoch Number: ", epoch, ", Loss: ", loss, ", Validation MAPE: ", accuracy, ".")
    


    