import torch.nn as nn
import torch
from torch_geometric.nn import GATv2Conv

class GATRNN(nn.Module):
    def __init__(self, num_static_node_features, num_static_edge_features, num_weather_features, hidden_size):
        super().__init__()
        self.gat_conv = GATv2Conv(num_static_node_features, hidden_size, edge_dim=num_static_edge_features)
        self.lstm = nn.LSTM(hidden_size + num_weather_features, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, node_static_features, edge_static_features, node_dynamic_features, edge_index):
        # Implement Later

        time_steps = node_dynamic_features.size(1)

        gat_output = self.gat_conv(node_static_features,edge_index.t(),edge_static_features)
        gat_output = torch.relu(gat_output)
        gat_output = gat_output.unsqueeze(1).expand(-1, time_steps, -1)
        rnn_input = torch.cat((gat_output, node_dynamic_features), dim=2)
        lstm_output, _ = self.lstm(rnn_input)
        output = self.linear(lstm_output)

        # Aggregate across time steps (mean pooling)
        mean_pooled_output = torch.mean(output, dim=1)
       
        return mean_pooled_output
