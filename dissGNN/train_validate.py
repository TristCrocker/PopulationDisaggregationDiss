import torch

def train(model, optimizer, data, loss_fn):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_weight)
    loss = loss_fn(output[data.train_mask].squeeze(), data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data):
    model.eval()

    with torch.no_grad():

        output = model(data.x, data.edge_index, data.edge_weight).squeeze()
        print("1.:", torch.expm1(output[data.val_mask]))
        print("2.:", torch.expm1(data.y[data.val_mask]))
        mae = (torch.expm1(output[data.val_mask]) - torch.expm1(data.y[data.val_mask])).abs().mean().item()
    return mae


def train_loop(num_epochs, model, data, optimizer, loss_fn):
    for epoch in range(num_epochs):
        loss = train(model, optimizer, data, loss_fn)
        accuracy = validate(model, data)
        print("Epoch Number: ", epoch, ", Loss: ", loss, ", Validation Accuracy: ", accuracy, ".")
    


    