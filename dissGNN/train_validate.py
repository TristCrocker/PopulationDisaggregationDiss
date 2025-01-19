def train(model, optimizer, data, loss_fn):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = loss_fn(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data):
    model.eval()
    output = model(data.x, data.edge_index)
    prediction = output.argmax(dim=1)
    correct = (prediction[data.val_mask] == data.y[data.val_mask]).sum()
    accuracy = int(correct) / int(data.val_mask.sum())
    return accuracy

def train_loop(num_epochs, model, data, optimizer, loss_fn):
    for epoch in range(num_epochs):
        loss = train(model, optimizer, data, loss_fn)
        accuracy = validate(model, data)

        if epoch % 10 == 0:
            print("Epoch Number: ", epoch, ", Loss: ", loss, ", Validation Accuracy: ", accuracy, ".")


    