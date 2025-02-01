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
    output = model(data.x, data.edge_index, data.edge_weight)
    prediction = torch.expm1(output.squeeze())
    # print(prediction)
    correct = torch.expm1(data.y)
    # print(correct)
    eps = 1e-6  # Small value to avoid division by zero
    error = torch.abs((prediction - correct) / (correct + eps))
    accuracy = 1 / (1 + error.mean().item()) 

    return accuracy

def train_loop(num_epochs, model, data, optimizer, loss_fn):
    for epoch in range(num_epochs):
        loss = train(model, optimizer, data, loss_fn)
        accuracy = validate(model, data)
        print("Epoch Number: ", epoch, ", Loss: ", loss, ", Validation Accuracy: ", accuracy, ".")
    


    