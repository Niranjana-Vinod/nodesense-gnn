import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNodeModel(nn.Module):
    """
    Graph Attention Network (Veličković et al., 2018).
    Learns dynamic attention coefficients between neighboring nodes.
    Supports returning attention weights for explainability (XAI).
    """
    def __init__(self, num_node_features: int, num_classes: int, hidden_dim: int = 8, heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, return_attention=False, return_embedding=False):
        if return_attention:
            emb, (edge_index_out, attn_weights) = self.conv1(x, edge_index, return_attention_weights=True)
        else:
            emb = self.conv1(x, edge_index)
            edge_index_out, attn_weights = None, None

        act_emb = F.elu(emb)
        act_emb = F.dropout(act_emb, p=self.dropout, training=self.training)
        out = self.conv2(act_emb, edge_index)

        if return_attention:
            return out, (edge_index_out, attn_weights)
        if return_embedding:
            return out, act_emb
        return out
