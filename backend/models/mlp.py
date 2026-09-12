import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBaseline(nn.Module):
    """
    Feature-only Multi-Layer Perceptron (MLP) baseline.
    Does NOT use graph topology, establishing an empirical baseline
    for what node features alone can predict without graph convolution.
    """
    def __init__(self, num_node_features: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(num_node_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index=None, return_embedding=False):
        emb = self.fc1(x)
        emb = F.relu(emb)
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        out = self.fc2(emb)
        if return_embedding:
            return out, emb
        return out
