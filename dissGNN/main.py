import torch.nn as nn
import model
import torch
import train_validate

if __name__ == "__main__":
    data = 0

    input_size = 0
    output_size = 0
    hidden_size = 0
    message_passing_count = 2
    drop_prob = 0.5
    learning_rate = 0.01

    model = model.GCN(input_size, output_size, hidden_size, message_passing_count, drop_prob)

    loss_fn = nn.CrossEntropLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)
    train_validate.train_loop(200, model, data, optimizer, loss_fn)
