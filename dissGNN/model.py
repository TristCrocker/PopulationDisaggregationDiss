import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GCN2Conv, GATv2Conv, TransformerConv
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch
import torch_geometric.utils
from torch_geometric.utils import add_self_loops

# Linear Regression Model
class LinRegModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinRegModel, self).__init__()
        # Linear layer
        self.linear = nn.Linear(input_size, output_size)

    # Forward
    def forward(self, x, edge_index=None, edge_weight=None):
        x = self.linear(x)
        return x

# GraphSage Model
class GraphSage(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size, message_passing_count, drop_prob):
        super(GraphSage, self).__init__()

        # Input to hidden layer
        self.conv = nn.ModuleList()
        self.conv.append(SAGEConv(input_size, hidden_layer_size)) 

        # Add number of hidden layers specified
        for i in range(message_passing_count - 1):
            self.conv.append(SAGEConv(hidden_layer_size, hidden_layer_size))

        # Hidden layer to output
        self.final_layer = SAGEConv(hidden_layer_size, output_size)

        # Dropout
        self.drop_prob = drop_prob

    # Forward
    def forward(self, x, edge_index, edge_weight):
        # Loop over all layers
        for layer in self.conv:
            x = layer(x, edge_index)        

            # Activation
            x = F.relu(x)

            # Dropout reg
            x = F.dropout(x, p=self.drop_prob, training = self.training)
        
        # Final Conv layer
        x = self.final_layer(x, edge_index)

        return x
    
# Graph Attention Network Model
class GAT(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size, message_passing_count, drop_prob):
        super(GAT, self).__init__()

        # Input to hidden layer
        self.conv = nn.ModuleList()
        self.conv.append(GATConv(input_size, hidden_layer_size, add_self_loops=True)) 
        
        # Add number of hidden layers specified
        for i in range(message_passing_count - 1):
            self.conv.append(GATConv(hidden_layer_size, hidden_layer_size, add_self_loops=True))

        # Hidden layer to output
        self.final_layer = GATConv(hidden_layer_size, output_size, add_self_loops=True)

        # Dropout
        self.drop_prob = drop_prob

    # Forward
    def forward(self, x, edge_index, edge_weight):
        # Loop over layers
        for layer in self.conv:
            x = layer(x, edge_index)        

            # Activation  
            x = F.relu(x)

            # Dropout reg
            x = F.dropout(x, p=self.drop_prob, training = self.training)
        
        # Final Conv layer
        x = self.final_layer(x, edge_index)
        return x
    
# Graph Convolutional Network Model
class GCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size, message_passing_count, drop_prob):
        super(GCN, self).__init__()

        # Input to hidden layer
        self.conv = nn.ModuleList()
        self.conv.append(GCNConv(input_size, hidden_layer_size, add_self_loops = True)) 
        
        # Add number of hidden layers specified
        for i in range(message_passing_count - 1):
            self.conv.append(GCNConv(hidden_layer_size, hidden_layer_size, add_self_loops=True))

        # Hidden layer to output
        self.final_layer = GCNConv(hidden_layer_size, output_size, add_self_loops=True)

        # Dropout
        self.drop_prob = drop_prob

    # Forward
    def forward(self, x, edge_index, edge_weight):
        # Loop over layers
        for layer in self.conv:
            x = layer(x, edge_index, edge_weight=edge_weight)   

            # Activation 
            x = F.relu(x)

            # Dropout reg
            x = F.dropout(x, p=self.drop_prob, training = self.training)

        # Final Conv layer
        x = self.final_layer(x, edge_index, edge_weight=edge_weight)

        return x
    
# Graph Convolutional Network Version 2 Model
class GCN2Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, drop_prob, alpha=0.1, theta=0.5):
        super().__init__()
        # Initial linear layer (Input to hidden layer)
        self.initial_proj = nn.Linear(input_size, hidden_size)

        # Add number of hidden layers specified
        self.conv = nn.ModuleList([
            GCN2Conv(hidden_size, alpha=alpha, theta=theta, layer=i + 1, add_self_loops=True)
            for i in range(num_layers)
        ])

        # Hidden layer to output
        self.final_proj = nn.Linear(hidden_size, output_size)

        # Dropout
        self.dropout = drop_prob

    # Forward
    def forward(self, x, edge_index, edge_weight):
        # Initial linear layer and store hidden rep for residuals
        x0 = x = self.initial_proj(x)

        # Loop over layers
        for layer in self.conv:
            x = layer(x, x0, edge_index)

            # Activation
            x = F.relu(x)

            # Dropout reg
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final Linear layer
        x = self.final_proj(x)
        return x
    
# Graph Convolutional Network Version 2 Model (With weights)
class GCN2NetWeights(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, drop_prob, alpha=0.1, theta=0.5):
        super().__init__()
        # Initial linear layer (Input to hidden layer)
        self.initial_proj = nn.Linear(input_size, hidden_size)

        # Add number of hidden layers specified
        self.conv = nn.ModuleList([
            GCN2Conv(hidden_size, alpha=alpha, theta=theta, layer=i + 1, add_self_loops=True)
            for i in range(num_layers)
        ])

        # Hidden layer to output
        self.final_proj = nn.Linear(hidden_size, output_size)

        # Dropout
        self.dropout = drop_prob

    # Forward
    def forward(self, x, edge_index, edge_weight):
        # Initial linear layer and store hidden rep for residuals
        x0 = x = self.initial_proj(x)

        # Loop over layers
        for layer in self.conv:
            x = layer(x, x0, edge_index, edge_weight=edge_weight)

            # Activation
            x = F.relu(x)

            # Dropout reg
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final linear layer
        x = self.final_proj(x)
        return x
    
# Graph Attention Network Version 2 Model
class GATv2Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, drop_prob, heads=4):
        super().__init__()

        # Input to hidden layer
        self.conv = nn.ModuleList()
        self.conv.append(GATv2Conv(input_size, hidden_size, heads=heads, add_self_loops=True))

        # Add number of hidden layers specified
        for _ in range(num_layers - 1):
            self.conv.append(GATv2Conv(hidden_size * heads, hidden_size, heads=heads, add_self_loops=True))

        #Hidden layer to output
        self.final = nn.Linear(hidden_size * heads, output_size)

        # Dropout
        self.dropout = drop_prob

    # Forward
    def forward(self, x, edge_index, edge_weight):
        # Loop over layers
        for layer in self.conv:
            x = layer(x, edge_index)

            # Activation
            x = F.relu(x)

            # Dropout reg
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final linear layer
        x = self.final(x)

        return x
    
# Graph Attention Network Version 2 Model (With weights)
class GATv2NetWeights(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, drop_prob, heads=4):
        super().__init__()
        # Input to hidden layer
        self.conv = nn.ModuleList()
        self.conv.append(GATv2Conv(input_size, hidden_size, heads=heads, edge_dim=1, add_self_loops=True))

        # Add number of hidden layers specified
        for _ in range(num_layers - 1):
            self.conv.append(GATv2Conv(hidden_size * heads, hidden_size, heads=heads, edge_dim=1, add_self_loops=True))

        # Hidden to output layer
        self.final = nn.Linear(hidden_size * heads, output_size)

        # Dropout
        self.dropout = drop_prob

    # Forward
    def forward(self, x, edge_index, edge_weight):
        # Preprocess edge_weights
        edge_attr = edge_weight.view(-1, 1) 

        # Loop over layers
        for layer in self.conv:
            x = layer(x, edge_index, edge_attr=edge_attr)

            # Activation
            x = F.relu(x)

            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final linear layer
        x = self.final(x)
        return x
    
# Transformer GNN Model
class TransformerNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, drop_prob, heads=4):
        super().__init__()
        # Input to hidden layer
        self.conv = nn.ModuleList()
        self.conv.append(TransformerConv(input_size, hidden_size, heads=heads))

        # Add number of hidden layers specified
        for _ in range(num_layers - 1):
            self.conv.append(TransformerConv(hidden_size * heads, hidden_size, heads=heads))

        # Hidden to output layer
        self.final = nn.Linear(hidden_size * heads, output_size)

        # Dropout
        self.dropout = drop_prob

    # Forward
    def forward(self, x, edge_index, edge_weight):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) 

        # Loop over layers
        for layer in self.conv:
            x = layer(x, edge_index)

            # Activation
            x = F.relu(x)

            # Dropout reg
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final linear layer
        x = self.final(x)
        return x