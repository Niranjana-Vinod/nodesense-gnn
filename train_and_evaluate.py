"""
NodeSense — Explainable and Robust GNNs for Scientific Citation Networks
End-to-End Training, Evaluation, Robustness Ablation, Link Prediction & XAI Script.
"""

import os
import sys
import time
import json
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

# Ensure backend modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from backend.data_loader import load_pubmed, get_graph_metadata, get_link_prediction_split, CLASS_LABELS
from backend.models.mlp import MLPBaseline
from backend.models.gcn import GCNNodeModel
from backend.models.gat import GATNodeModel
from backend.models.graphsage import GraphSAGENodeModel
from backend.models.link_predictor import DotProductLinkPredictor, evaluate_link_prediction

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_single_model(model, data, epochs=100, lr=0.01, weight_decay=5e-4, device='cpu', is_mlp=False):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    start_time = time.time()
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        if is_mlp:
            out = model(data.x)
        else:
            out = model(data.x, data.edge_index)
            
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation tracking
        model.eval()
        with torch.no_grad():
            if is_mlp:
                val_out = model(data.x)
            else:
                val_out = model(data.x, data.edge_index)
            val_pred = val_out[data.val_mask].argmax(dim=1)
            val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    train_time = time.time() - start_time
    
    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, train_time

def evaluate_metrics(model, data, device='cpu', is_mlp=False):
    model.eval()
    with torch.no_grad():
        if is_mlp:
            out = model(data.x.to(device))
        else:
            out = model(data.x.to(device), data.edge_index.to(device))
            
        preds = out[data.test_mask].argmax(dim=1).cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()
        
    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    macro_prec = precision_score(y_true, preds, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, preds, average='macro', zero_division=0)
    
    return {
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "precision": float(round(macro_prec, 4)),
        "recall": float(round(macro_rec, 4))
    }

def run_edge_dropout_ablation(model, data, drop_rates=[0.0, 0.2, 0.5], num_trials=3, device='cpu', is_mlp=False):
    results = {}
    model.eval()
    with torch.no_grad():
        for drop in drop_rates:
            trial_accs = []
            for t in range(num_trials):
                if is_mlp or drop == 0.0:
                    eval_edges = data.edge_index.to(device)
                else:
                    # Randomly drop edges
                    mask = torch.rand(data.edge_index.size(1)) > drop
                    eval_edges = data.edge_index[:, mask].to(device)
                    
                if is_mlp:
                    out = model(data.x.to(device))
                else:
                    out = model(data.x.to(device), eval_edges)
                    
                preds = out[data.test_mask].argmax(dim=1).cpu().numpy()
                y_true = data.y[data.test_mask].cpu().numpy()
                trial_accs.append(accuracy_score(y_true, preds))
            results[str(drop)] = float(round(float(np.mean(trial_accs)), 4))
    return results

def main():
    parser = argparse.ArgumentParser(description="NodeSense GNN Benchmark & Training")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per model")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--data_dir", type=str, default="data/PubMed", help="Dataset directory")
    parser.add_argument("--checkpoints_dir", type=str, default="backend/checkpoints", help="Checkpoints save directory")
    parser.add_argument("--results_dir", type=str, default="backend/results", help="Results save directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 70)
    print("NodeSense — Explainable and Robust GNNs for Scientific Citation Networks")
    print(f"Device: {args.device} | Epochs: {args.epochs} | LR: {args.lr}")
    print("=" * 70)

    # -------------------------------------------------------------
    # PHASE 1: Load and Validate PubMed Dataset
    # -------------------------------------------------------------
    print("\n>>> [PHASE 1] Loading and Validating PubMed...")
    dataset, data = load_pubmed(root_dir=args.data_dir, seed=42)
    meta = get_graph_metadata(dataset, data)
    print(f"Nodes: {meta['num_nodes']}, Edges: {meta['num_edges']}, Features: {meta['num_features']}, Classes: {meta['num_classes']}")
    print(f"Split -> Train: {meta['train_nodes']}, Val: {meta['val_nodes']}, Test: {meta['test_nodes']}")
    
    with open(os.path.join(args.results_dir, "graph_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    data_dev = data.to(args.device)
    num_features = dataset.num_node_features
    num_classes = dataset.num_classes

    # Define the 4 Core Models
    models_dict = {
        "MLP (Baseline)": {
            "instance": MLPBaseline(num_features, num_classes, hidden_dim=64),
            "is_mlp": True,
            "filename": "mlp_baseline.pt"
        },
        "GCN": {
            "instance": GCNNodeModel(num_features, num_classes, hidden_dim=64),
            "is_mlp": False,
            "filename": "gcn_model.pt"
        },
        "GAT": {
            "instance": GATNodeModel(num_features, num_classes, hidden_dim=8, heads=8),
            "is_mlp": False,
            "filename": "gat_model.pt"
        },
        "GraphSAGE": {
            "instance": GraphSAGENodeModel(num_features, num_classes, hidden_dim=64),
            "is_mlp": False,
            "filename": "sage_model.pt"
        }
    }

    benchmark_records = []

    print("\n>>> [PHASE 1] Training and Evaluating Core Models...")
    for name, info in models_dict.items():
        print(f"\n--- Training {name} ---")
        trained_model, t_time = train_single_model(
            model=info["instance"],
            data=data_dev,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            is_mlp=info["is_mlp"]
        )
        # Save checkpoint
        ckpt_path = os.path.join(args.checkpoints_dir, info["filename"])
        torch.save(trained_model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to: {ckpt_path}")
        
        # Evaluate on test set
        metrics = evaluate_metrics(trained_model, data, device=args.device, is_mlp=info["is_mlp"])
        metrics["model"] = name
        metrics["training_time_sec"] = round(t_time, 2)
        metrics["beats_baseline"] = False
        print(f"Metrics: Acc={metrics['accuracy']:.4f} | F1={metrics['macro_f1']:.4f} | Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f} | Time={metrics['training_time_sec']}s")
        benchmark_records.append(metrics)

    # Determine baseline comparison
    mlp_acc = benchmark_records[0]["accuracy"]
    for rec in benchmark_records:
        if rec["model"] != "MLP (Baseline)":
            rec["beats_baseline"] = bool(rec["accuracy"] > mlp_acc)
            rec["delta_vs_baseline"] = round(rec["accuracy"] - mlp_acc, 4)
        else:
            rec["beats_baseline"] = None
            rec["delta_vs_baseline"] = 0.0

    # Save Phase 1 Benchmark Results
    bench_json_path = os.path.join(args.results_dir, "benchmark_results.json")
    with open(bench_json_path, "w") as f:
        json.dump(benchmark_records, f, indent=2)
        
    bench_csv_path = os.path.join(args.results_dir, "benchmark_results.csv")
    with open(bench_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "macro_f1", "precision", "recall", "training_time_sec", "beats_baseline", "delta_vs_baseline"])
        writer.writeheader()
        writer.writerows(benchmark_records)
    print(f"\nPhase 1 Results saved to {bench_json_path} and {bench_csv_path}")

    # -------------------------------------------------------------
    # PHASE 2: Robustness — Edge-Dropout Ablation (0%, 20%, 50%)
    # -------------------------------------------------------------
    print("\n>>> [PHASE 2] Running Robustness Stress Test (Edge Dropout: 0%, 20%, 50%)...")
    drop_rates = [0.0, 0.2, 0.5]
    robustness_data = {}
    
    for name, info in models_dict.items():
        print(f"Testing robustness for {name}...")
        drop_res = run_edge_dropout_ablation(
            model=info["instance"],
            data=data,
            drop_rates=drop_rates,
            num_trials=3,
            device=args.device,
            is_mlp=info["is_mlp"]
        )
        robustness_data[name] = drop_res
        print(f"  {name} Accuracies: drop 0%={drop_res['0.0']}, drop 20%={drop_res['0.2']}, drop 50%={drop_res['0.5']}")

    # Save Phase 2 Robustness Results
    rob_json_path = os.path.join(args.results_dir, "robustness_results.json")
    with open(rob_json_path, "w") as f:
        json.dump(robustness_data, f, indent=2)
        
    rob_csv_path = os.path.join(args.results_dir, "robustness_results.csv")
    with open(rob_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Drop_0.0", "Drop_0.2", "Drop_0.5"])
        for m_name, d_res in robustness_data.items():
            writer.writerow([m_name, d_res["0.0"], d_res["0.2"], d_res["0.5"]])
    print(f"Phase 2 Robustness saved to {rob_json_path} and {rob_csv_path}")

    # -------------------------------------------------------------
    # PHASE 3: Link Prediction / Citation Recommendation
    # -------------------------------------------------------------
    print("\n>>> [PHASE 3] Evaluating Link Prediction on GNN Embeddings...")
    test_pos_edges, test_neg_edges = get_link_prediction_split(data, test_ratio=0.10, seed=42)
    test_pos_edges = test_pos_edges.to(args.device)
    test_neg_edges = test_neg_edges.to(args.device)

    link_results = {}
    embedding_cache = {}

    for name, info in models_dict.items():
        if info["is_mlp"]:
            continue
        m = info["instance"]
        m.eval()
        with torch.no_grad():
            _, emb = m(data_dev.x, data_dev.edge_index, return_embedding=True)
            auc, ap, _, _ = evaluate_link_prediction(emb, test_pos_edges, test_neg_edges)
            link_results[name] = {
                "roc_auc": round(auc, 4),
                "average_precision": round(ap, 4)
            }
            embedding_cache[name] = emb.cpu()
            print(f"  {name} Link Prediction -> ROC-AUC: {auc:.4f} | AP: {ap:.4f}")

    link_json_path = os.path.join(args.results_dir, "link_prediction_results.json")
    with open(link_json_path, "w") as f:
        json.dump(link_results, f, indent=2)
    print(f"Phase 3 Link Prediction saved to {link_json_path}")

    # -------------------------------------------------------------
    # PHASE 4: Explainability (XAI) — GAT Attention Weights
    # -------------------------------------------------------------
    print("\n>>> [PHASE 4] Extracting GAT Multi-Head Attention Weights...")
    gat_m = models_dict["GAT"]["instance"]
    gat_m.eval()
    with torch.no_grad():
        _, (edge_index_out, attn_weights) = gat_m(data_dev.x, data_dev.edge_index, return_attention=True)
        # Average attention weights across 8 heads
        avg_attn = attn_weights.mean(dim=-1).squeeze().cpu().numpy()
        edge_index_cpu = edge_index_out.cpu().numpy()

    # Find top 10 most attended citation links in test set
    top_indices = avg_attn.argsort()[-15:][::-1]
    sample_explanations = []
    for idx in top_indices:
        src = int(edge_index_cpu[0, idx])
        dst = int(edge_index_cpu[1, idx])
        w = float(round(float(avg_attn[idx]), 4))
        src_cls = int(data.y[src].item())
        dst_cls = int(data.y[dst].item())
        sample_explanations.append({
            "source_node": src,
            "target_node": dst,
            "attention_weight": w,
            "source_class": CLASS_LABELS.get(src_cls, f"Class {src_cls}"),
            "target_class": CLASS_LABELS.get(dst_cls, f"Class {dst_cls}"),
            "same_topic": bool(src_cls == dst_cls)
        })

    xai_json_path = os.path.join(args.results_dir, "xai_attention_sample.json")
    with open(xai_json_path, "w") as f:
        json.dump(sample_explanations, f, indent=2)
    print(f"Phase 4 XAI Explanations saved to {xai_json_path}")
    print("\nTop 5 GAT Citation Attentions:")
    for item in sample_explanations[:5]:
        print(f"  Paper #{item['source_node']} -> Paper #{item['target_node']} | Attn: {item['attention_weight']} | Same Topic: {item['same_topic']}")

    print("\n" + "=" * 70)
    print("ALL PHASES 1-4 COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
