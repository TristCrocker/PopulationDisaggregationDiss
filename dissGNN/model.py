import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import torch

class GCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size, message_passing_count, drop_prob):
        super(GCN, self).__init__()

        #Input to hidden layer mapping
        self.conv = nn.ModuleList()
        self.conv.append(GCNConv(input_size, hidden_layer_size, add_self_loops=True)) 
        
        #Hidden layer to hidden layer mapping
        #Add number of hidden layers to match the number of neighbours
        for i in range(message_passing_count - 1):
            self.conv.append(GCNConv(hidden_layer_size, hidden_layer_size, add_self_loops=True))

        #Hidden layer to output mapping
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.drop_prob = drop_prob


    def forward(self, x, edge_index, edge_weight):

        for layer in self.conv:
            x = F.dropout(x, p=self.drop_prob, training = self.training)
            x = layer(x, edge_index, edge_weight=edge_weight)          
            x = F.relu(x)

        x = self.linear(x)

        return x