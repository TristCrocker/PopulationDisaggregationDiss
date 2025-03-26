import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GCN2Conv
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch


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
        self.conv.append(GATConv(input_size, hidden_layer_size, add_self_loops=True, edge_dim=1)) 
        
        #Hidden layer to hidden layer mapping
        #Add number of hidden layers to match the number of neighbours
        for i in range(message_passing_count - 1):
            self.conv.append(GATConv(hidden_layer_size, hidden_layer_size, add_self_loops=True, edge_dim=1))

        #Hidden layer to output mapping
        self.final_layer = GATConv(hidden_layer_size, output_size, add_self_loops=True, edge_dim=1)
        self.drop_prob = drop_prob

    def forward(self, x, edge_index, edge_weight):
        edge_weight = edge_weight.view(-1, 1)
        # resid = x
        for layer in self.conv:
            x = layer(x, edge_index, edge_attr=edge_weight)          
            # x = F.leaky_relu(x, negative_slope = 0.1)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_prob, training = self.training)

            # if x.shape[1] != resid.shape[1]:  
            #     resid = torch.nn.Linear(resid.shape[1], x.shape[1]).to(x.device)(resid)
            
            # x = resid + x
            # resid = x
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
            x = layer(x, edge_index, edge_weight=edge_weight)      
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_prob, training = self.training)

        x = self.final_layer(x, edge_index, edge_weight=edge_weight)

        return x