import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size, message_passing_count, drop_prob):
        super().__init__()

        #Input to hidden layer mapping
        self.conv = nn.ModuleList()
        self.conv.append(GCNConv(input_size, hidden_layer_size, add_self_loops=True)) 
        
        #Hidden layer to hidden layer mapping
        #Add number of hidden layers to match the number of neighbours
        for i in range(message_passing_count):
            self.conv.append(GCNConv(hidden_layer_size, hidden_layer_size, add_self_loops=True))

        #Hidden layer to output mapping
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.drop_prob = drop_prob


    def forward_pass(self, x, edge_index):
        for layer in self.conv:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_prob, training = self.training)

        x = self.linear(x)

        return x