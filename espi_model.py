import torch
import torch.nn as nn
from torch.nn import GRUCell
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


class GGNNLayer(MessagePassing):
    def __init__(self, hidden_size):
        super(GGNNLayer, self).__init__(aggr="add")
        self.gru = GRUCell(hidden_size, hidden_size)

    def forward(self, x, edge_index):
        x_neighbor = self.propagate(edge_index, x=x)
        x = self.gru(x_neighbor, x)
        return x


class GGNN(nn.Module):
    def __init__(self, hidden_size, layer_num, dropout=0.1) -> None:
        super().__init__()
        assert layer_num > 0
        self.mod_list = nn.ModuleList(
            [GGNNLayer(hidden_size)]
            + [GGNNLayer(hidden_size) for _ in range(layer_num - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, edge_index):
        for m in self.mod_list:
            x = m(x, edge_index)
        x = self.dropout(x)
        x = self.dense(x)
        x = x.max(dim=0).values
        return x


class ESPI_MSG_MODEL(nn.Module):
    def __init__(self, hidden_size, layer_num, dropout=0.1) -> None:
        super().__init__()
        self.ggnn = GGNN(hidden_size, layer_num, dropout)
        self.clf = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.ggnn(x, edge_index)
        x = self.dropout(x)
        x = self.clf(x)
        x = torch.sigmoid(x)
        return x
