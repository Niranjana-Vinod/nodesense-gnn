import os
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph, negative_sampling

CLASS_LABELS = {
    0: "Experimental Diabetes",
    1: "Type 1 Diabetes",
    2: "Type 2 Diabetes"
}

CLASS_COLORS = {
    0: "#3b82f6",  # Blue
    1: "#10b981",  # Emerald green
    2: "#f59e0b"   # Amber
}

def load_pubmed(root_dir: str = 'data/PubMed', seed: int = 42):
    """
    Loads PubMed citation dataset with reproducible seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(root_dir, exist_ok=True)
    dataset = Planetoid(root=root_dir, name='PubMed')
    data = dataset[0]
    return dataset, data

def get_graph_metadata(dataset, data):
    """
    Returns summary metrics of the PubMed graph.
    """
    train_count = int(data.train_mask.sum().item())
    val_count = int(data.val_mask.sum().item())
    test_count = int(data.test_mask.sum().item())
    
    deg = torch.bincount(data.edge_index[0], minlength=data.num_nodes).float()
    avg_degree = float(deg.mean().item())
    max_degree = int(deg.max().item())
    
    class_counts = torch.bincount(data.y, minlength=dataset.num_classes).tolist()
    
    return {
        "dataset_name": "PubMed (Diabetes Citation Network)",
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(dataset.num_node_features),
        "num_classes": int(dataset.num_classes),
        "train_nodes": train_count,
        "val_nodes": val_count,
        "test_nodes": test_count,
        "avg_degree": round(avg_degree, 2),
        "max_degree": max_degree,
        "class_distribution": {
            CLASS_LABELS.get(i, f"Class {i}"): class_counts[i] for i in range(len(class_counts))
        }
    }

def get_node_neighbors(data, node_id: int):
    """
    Returns the set of 1-hop neighbor node IDs for a given node.
    """
    mask = data.edge_index[0] == node_id
    neighbors = data.edge_index[1][mask].tolist()
    return set(neighbors)

def get_ego_subgraph(data, center_node: int, num_hops: int = 1, max_nodes: int = 60):
    """
    Extracts k-hop ego subgraph around center_node for fast, interactive visualizer rendering.
    """
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=center_node,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )
    
    # If subgraph is too large for smooth Canvas rendering, sample top degree neighbors
    if len(subset) > max_nodes:
        # Keep center node + top (max_nodes - 1) neighbors
        sub_list = subset.tolist()
        center_orig = sub_list[mapping.item()]
        other_nodes = [n for n in sub_list if n != center_orig]
        # pick a subset
        np.random.seed(42)
        sampled = [center_orig] + list(np.random.choice(other_nodes, max_nodes - 1, replace=False))
        sampled_set = set(sampled)
        
        # Filter original edges
        mask_u = [u.item() in sampled_set for u in data.edge_index[0]]
        mask_v = [v.item() in sampled_set for v in data.edge_index[1]]
        both_mask = torch.tensor([a and b for a, b in zip(mask_u, mask_v)], dtype=torch.bool)
        
        sub_edges = data.edge_index[:, both_mask]
        node_to_idx = {nid: idx for idx, nid in enumerate(sampled)}
        
        nodes_out = []
        for nid in sampled:
            nid_int = int(nid)
            cls = int(data.y[nid_int].item())
            nodes_out.append({
                "id": nid_int,
                "label": f"Paper #{nid_int}",
                "class_id": cls,
                "class_name": CLASS_LABELS.get(cls, f"Class {cls}"),
                "color": CLASS_COLORS.get(cls, "#6366f1"),
                "is_center": True if nid_int == int(center_node) else False,
                "is_train": True if bool(data.train_mask[nid_int].item()) else False,
                "is_test": True if bool(data.test_mask[nid_int].item()) else False
            })
            
        edges_out = []
        for i in range(sub_edges.size(1)):
            src = int(sub_edges[0, i].item())
            dst = int(sub_edges[1, i].item())
            edges_out.append({
                "source": src,
                "target": dst
            })
        return {"nodes": nodes_out, "edges": edges_out, "center_id": int(center_node)}
    
    # Normal return when <= max_nodes
    nodes_out = []
    subset_list = subset.tolist()
    for local_idx, orig_id in enumerate(subset_list):
        nid_int = int(orig_id)
        cls = int(data.y[nid_int].item())
        nodes_out.append({
            "id": nid_int,
            "label": f"Paper #{nid_int}",
            "class_id": cls,
            "class_name": CLASS_LABELS.get(cls, f"Class {cls}"),
            "color": CLASS_COLORS.get(cls, "#6366f1"),
            "is_center": True if nid_int == int(center_node) else False,
            "is_train": True if bool(data.train_mask[nid_int].item()) else False,
            "is_test": True if bool(data.test_mask[nid_int].item()) else False
        })

        
    edges_out = []
    for i in range(edge_index.size(1)):
        src_local = int(edge_index[0, i].item())
        dst_local = int(edge_index[1, i].item())
        edges_out.append({
            "source": subset_list[src_local],
            "target": subset_list[dst_local]
        })
        
    return {"nodes": nodes_out, "edges": edges_out, "center_id": center_node}

def get_link_prediction_split(data, test_ratio: float = 0.10, seed: int = 42):
    """
    Extracts positive test edges and generates equal number of negative (non-edge) samples.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    num_edges = data.edge_index.size(1)
    num_test = int(num_edges * test_ratio)
    
    perm = torch.randperm(num_edges)
    test_pos_edge_index = data.edge_index[:, perm[:num_test]]
    
    # Negative sampling: random pairs of nodes that do not have an edge
    test_neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=num_test,
        method='sparse'
    )
    
    return test_pos_edge_index, test_neg_edge_index
