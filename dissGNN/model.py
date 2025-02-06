import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch.nn.functional as F
import torch

class GraphSage(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size, message_passing_count, drop_prob):
        super(GraphSage, self).__init__()

        #Input to hidden layer mapping
        self.conv = nn.ModuleList()
        self.conv.append(SAGEConv(input_size, hidden_layer_size)) 
        
        #Hidden layer to hidden layer mapping
        #Add number of hidden layers to match the number of neighbours
        for i in range(message_passing_count - 1):
            self.conv.append(SAGEConv(hidden_layer_size, hidden_layer_size))

        #Hidden layer to output mapping
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.drop_prob = drop_prob

    def forward(self, x, edge_index, edge_weight):

        row, col = edge_index  # Extract source & target nodes

        # Check if edge_weight exists, otherwise default to 1
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=x.device)

        # Normalize edge weights (Optional, can be removed)
        edge_weight = edge_weight / (edge_weight.max() + 1e-6)  

        for conv in self.convs:
            x = F.dropout(x, p=self.drop_prob, training=self.training)

            # Aggregate weighted node features using scatter_add
            weighted_x = x[col] * edge_weight.view(-1, 1)  # Apply edge weights
            x_agg = scatter_add(weighted_x, row, dim=0, dim_size=x.size(0))  # Sum messages per node

            x = conv(x_agg, edge_index)  # Pass through GraphSAGE layer
            x = F.relu(x)

        x = self.linear(x)  # Final layer mapping

        return x
    
class GCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size, message_passing_count, drop_prob):
        super(GCN, self).__init__()

        #Input to hidden layer mapping
        self.conv = nn.ModuleList()
        self.conv.append(GCNConv(input_size, hidden_layer_size, add_self_loops = True)) 
        
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
            # x = F.leaky_relu(x, negative_slope = 0.01)
            x = F.relu6(x)

        x = self.linear(x)

        return x