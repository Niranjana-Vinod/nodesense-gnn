import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

class DotProductLinkPredictor(nn.Module):
    """
    Dot-product link predictor operating on GNN node embeddings.
    Calculates probability P(u, v is edge) = sigmoid(z_u^T * z_v).
    """
    def __init__(self):
        super().__init__()

    def forward(self, z, edge_index):
        src, dst = edge_index[0], edge_index[1]
        score = (z[src] * z[dst]).sum(dim=-1)
        return torch.sigmoid(score)

    def predict_pair(self, z, u: int, v: int) -> float:
        score = (z[u] * z[v]).sum()
        return float(torch.sigmoid(score).item())

    def recommend_top_k(self, z, query_node: int, existing_neighbors: set, k: int = 5, candidate_nodes: list = None):
        """
        Given a query paper node, find top-k non-connected papers with highest link likelihood.
        """
        num_nodes = z.size(0)
        q_emb = z[query_node].unsqueeze(0)  # [1, D]
        
        if candidate_nodes is not None:
            cands = [c for c in candidate_nodes if c != query_node and c not in existing_neighbors]
            if not cands:
                return []
            cand_tensor = torch.tensor(cands, dtype=torch.long, device=z.device)
            cand_embs = z[cand_tensor]  # [N_c, D]
            scores = torch.sigmoid((q_emb * cand_embs).sum(dim=-1)).cpu().numpy()
            top_indices = np.argsort(scores)[::-1][:k]
            return [(int(cands[i]), float(scores[i])) for i in top_indices]
        else:
            # Full node scoring
            scores = torch.sigmoid((q_emb * z).sum(dim=-1)).squeeze().cpu().numpy()
            # Mask self and existing neighbors
            scores[query_node] = -1.0
            for n in existing_neighbors:
                if n < len(scores):
                    scores[n] = -1.0
            top_nodes = np.argsort(scores)[::-1][:k]
            return [(int(node_id), float(scores[node_id])) for node_id in top_nodes]

def evaluate_link_prediction(z, pos_edge_index, neg_edge_index):
    """
    Evaluates link prediction given positive and negative edge samples.
    Returns (roc_auc, average_precision, pos_scores, neg_scores).
    """
    predictor = DotProductLinkPredictor()
    with torch.no_grad():
        pos_preds = predictor(z, pos_edge_index).cpu().numpy()
        neg_preds = predictor(z, neg_edge_index).cpu().numpy()

    y_true = np.concatenate([np.ones_like(pos_preds), np.zeros_like(neg_preds)])
    y_scores = np.concatenate([pos_preds, neg_preds])

    auc = float(roc_auc_score(y_true, y_scores))
    ap = float(average_precision_score(y_true, y_scores))
    return auc, ap, pos_preds, neg_preds
