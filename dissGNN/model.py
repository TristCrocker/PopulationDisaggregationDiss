import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GCN2Conv, APPNP, GATv2Conv, ResGatedGraphConv, TransformerConv
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch
import torch_geometric.utils


class LinRegModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinRegModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, edge_index=None, edge_weight=None):
        x = self.linear(x)
        return x

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
        self.final_layer = SAGEConv(hidden_layer_size, output_size)
        self.drop_prob = drop_prob

    def forward(self, x, edge_index, edge_weight):
        for layer in self.conv:
            x = F.dropout(x, p=self.drop_prob, training = self.training)
            x = layer(x, edge_index)          
            # x = F.leaky_relu(x, negative_slope = 0.01)
            x = F.relu(x)

        x = self.final_layer(x, edge_index)

        return x
    
class GAT(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size, message_passing_count, drop_prob):
        super(GAT, self).__init__()

        #Input to hidden layer mapping
        self.conv = nn.ModuleList()
        self.conv.append(GATConv(input_size, hidden_layer_size, add_self_loops=True)) 
        
        #Hidden layer to hidden layer mapping
        #Add number of hidden layers to match the number of neighbours
        for i in range(message_passing_count - 1):
            self.conv.append(GATConv(hidden_layer_size, hidden_layer_size, add_self_loops=True))

        #Hidden layer to output mapping
        self.final_layer = GATConv(hidden_layer_size, output_size, add_self_loops=True)
        self.drop_prob = drop_prob

    def forward(self, x, edge_index, edge_weight):
        # resid = x
        for layer in self.conv:
            residual = x
            x = F.dropout(x, p=self.drop_prob, training = self.training)
            x = layer(x, edge_index, edge_attr=edge_weight)          
            # x = F.leaky_relu(x, negative_slope = 0.1)

            if x.shape == residual.shape:
                x = x + residual  
            x = F.relu(x)
            

        x = self.final_layer(x, edge_index, edge_attr=edge_weight)
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
        self.final_layer = GCNConv(hidden_layer_size, output_size, add_self_loops=True)
        self.drop_prob = drop_prob

    def forward(self, x, edge_index, edge_weight):
        for layer in self.conv:
            residual = x
            x = F.dropout(x, p=self.drop_prob, training = self.training)
            x = layer(x, edge_index, edge_weight=edge_weight)    
            if x.shape == residual.shape:
                x = x + residual  
            x = F.relu(x)

        x = self.final_layer(x, edge_index, edge_weight=edge_weight)

        return x
    
class GCN2Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, drop_prob, alpha=0.1, theta=0.5):
        super().__init__()
        self.initial_proj = nn.Linear(input_size, hidden_size)
        self.gcn_layers = nn.ModuleList([
            GCN2Conv(hidden_size, alpha=alpha, theta=theta, layer=i + 1)
            for i in range(num_layers)
        ])
        self.final_proj = nn.Linear(hidden_size, output_size)
        self.dropout = drop_prob

    def forward(self, x, edge_index, edge_weight):
        x0 = x = self.initial_proj(x)  # project input to hidden size
        for layer in self.gcn_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, x0, edge_index)
            x = F.relu(x)

        x = self.final_proj(x)
        return x
    
class GATv2Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, drop_prob, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATv2Conv(input_size, hidden_size, heads=heads))
        for _ in range(num_layers - 1):
            self.layers.append(GATv2Conv(hidden_size * heads, hidden_size, heads=heads))
        self.final = nn.Linear(hidden_size * heads, output_size)
        self.dropout = drop_prob

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        x = self.final(x)
        return x
    

class TransformerNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, drop_prob, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TransformerConv(input_size, hidden_size, heads=heads))
        for _ in range(num_layers - 1):
            self.layers.append(TransformerConv(hidden_size * heads, hidden_size, heads=heads))
        self.final = nn.Linear(hidden_size * heads, output_size)
        self.dropout = drop_prob

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, edge_index))
        x = self.final(x)
        return x