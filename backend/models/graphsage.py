import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGENodeModel(nn.Module):
    """
    GraphSAGE (Hamilton et al., 2017).
    Inductive representation learning via uniform neighborhood sampling & aggregation.
    """
    def __init__(self, num_node_features: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, return_embedding=False):
        emb = self.conv1(x, edge_index)
        emb = F.relu(emb)
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        out = self.conv2(emb, edge_index)
        if return_embedding:
            return out, emb
        return out
