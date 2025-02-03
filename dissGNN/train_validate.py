import torch

def train(model, optimizer, data, loss_fn):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_weight)
    loss = loss_fn(torch.expm1(output[data.train_mask].squeeze()), torch.expm1(data.y[data.train_mask]))
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data):
    model.eval()

    with torch.no_grad():

        output = model(data.x, data.edge_index, data.edge_weight).squeeze()
        # mae = (torch.expm1(output[data.val_mask]) - torch.expm1(data.y[data.val_mask])).abs().mean().item()
        mae = (torch.expm1(output[data.val_mask]) - torch.expm1(data.y[data.val_mask])).abs().mean().item()
        # rmse = torch.sqrt(((torch.expm1(output[data.train_mask]) - torch.expm1(data.y[data.train_mask])) ** 2).mean()).item()
    return mae


def train_loop(num_epochs, model, data, optimizer, loss_fn):
    for epoch in range(num_epochs):
        loss = train(model, optimizer, data, loss_fn)
        accuracy = validate(model, data)
        print("Epoch Number: ", epoch, ", Loss: ", loss, ", Validation MAE: ", accuracy, ".")
    


    