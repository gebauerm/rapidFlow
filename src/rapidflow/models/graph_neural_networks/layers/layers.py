import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling


class InnerProductLayer(nn.Module):

    def __init__(self, dropout=0):
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout_layer(x)
        return torch.matmul(x, x.t())


class SparseInnerProductLayer(nn.Module):
    def __init__(self, negative_sampling_num=None):
        super().__init__()
        self.negative_sampling_num = negative_sampling_num

    def forward(self, x, edge_index):
        positive_edges = torch.sum(x[edge_index[0]]*x[edge_index[1]], dim=1)
        negative_edges = torch.tensor([[0]])
        if self.negative_sampling_num:
            negative_edge_index = negative_sampling(edge_index, self.negative_sampling_num)
            negative_edges = torch.sum(x[negative_edge_index[0]]*x[negative_edge_index[1]], dim=1)
        return positive_edges, negative_edges


class ProductLayer(nn.Module):

    def __init__(self, dropout=0):
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, y):
        x = self.dropout_layer(x)
        return torch.matmul(x, y.t())
