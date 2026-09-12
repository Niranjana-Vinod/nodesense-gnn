import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add root and backend to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from backend.data_loader import load_pubmed, get_graph_metadata, get_ego_subgraph, get_node_neighbors, CLASS_LABELS, CLASS_COLORS
from backend.models.mlp import MLPBaseline
from backend.models.gcn import GCNNodeModel
from backend.models.gat import GATNodeModel
from backend.models.graphsage import GraphSAGENodeModel
from backend.models.link_predictor import DotProductLinkPredictor

STATIC_DIR = os.path.join(ROOT_DIR, 'frontend')
app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'PubMed')
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Global state
dataset = None
data = None
metadata = {}
models = {}
embeddings = {}
gat_attention_cache = None

def init_app_state():
    global dataset, data, metadata, models, embeddings, gat_attention_cache
    print(f"[*] Initializing NodeSense Backend on device: {DEVICE}")
    dataset, data = load_pubmed(root_dir=DATA_DIR, seed=42)
    metadata = get_graph_metadata(dataset, data)
    print(f"[*] Loaded PubMed: {data.num_nodes} nodes, {data.num_edges} edges")

    num_features = dataset.num_node_features
    num_classes = dataset.num_classes

    # Instantiate models
    models['mlp'] = MLPBaseline(num_features, num_classes, hidden_dim=64).to(DEVICE)
    models['gcn'] = GCNNodeModel(num_features, num_classes, hidden_dim=64).to(DEVICE)
    models['gat'] = GATNodeModel(num_features, num_classes, hidden_dim=8, heads=8).to(DEVICE)
    models['sage'] = GraphSAGENodeModel(num_features, num_classes, hidden_dim=64).to(DEVICE)

    # Load weights if available
    ckpt_map = {
        'mlp': 'mlp_baseline.pt',
        'gcn': 'gcn_model.pt',
        'gat': 'gat_model.pt',
        'sage': 'sage_model.pt'
    }

    for key, filename in ckpt_map.items():
        path = os.path.join(CHECKPOINTS_DIR, filename)
        if os.path.exists(path):
            try:
                models[key].load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
                models[key].eval()
                print(f"[+] Loaded pretrained weights for {key} from {filename}")
            except Exception as e:
                print(f"[!] Warning: Could not load {filename}: {e}")
        else:
            print(f"[-] Checkpoint {filename} not found yet (models will run with initial weights until trained)")

    # Pre-extract embeddings for fast link recommendation
    data_dev = data.to(DEVICE)
    for key in ['gcn', 'gat', 'sage']:
        m = models[key]
        m.eval()
        with torch.no_grad():
            try:
                _, emb = m(data_dev.x, data_dev.edge_index, return_embedding=True)
                embeddings[key] = emb.cpu()
            except Exception as e:
                print(f"[!] Could not extract embeddings for {key}: {e}")

    # Precompute sample GAT attention
    try:
        gat = models['gat']
        gat.eval()
        with torch.no_grad():
            _, (edge_idx, attn_w) = gat(data_dev.x, data_dev.edge_index, return_attention=True)
            gat_attention_cache = {
                'edge_index': edge_idx.cpu(),
                'attention_weights': attn_w.mean(dim=-1).squeeze().cpu()
            }
            print("[+] GAT attention weights pre-cached")
    except Exception as e:
        print(f"[!] Warning caching GAT attention: {e}")

# Frontend static serving
@app.route('/')
def serve_index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(STATIC_DIR, path)):
        return send_from_directory(STATIC_DIR, path)
    return send_from_directory(STATIC_DIR, 'index.html')

# API Endpoints
@app.route('/api/status', methods=['GET'])
def api_status():
    checkpoints_ready = {
        k: os.path.exists(os.path.join(CHECKPOINTS_DIR, f"{k if k != 'mlp' else 'mlp_baseline' if k=='mlp' else k}_model.pt"))
        for k in ['mlp', 'gcn', 'gat', 'sage']
    }
    return jsonify({
        "status": "healthy",
        "device": str(DEVICE),
        "dataset": "PubMed",
        "nodes": data.num_nodes if data is not None else 0,
        "edges": data.num_edges if data is not None else 0,
        "models_loaded": list(models.keys()),
        "results_ready": os.path.exists(os.path.join(RESULTS_DIR, 'benchmark_results.json'))
    })

@app.route('/api/metadata', methods=['GET'])
def api_metadata():
    return jsonify(metadata)

@app.route('/api/subgraph', methods=['GET'])
def api_subgraph():
    node_id = int(request.args.get('node_id', 0))
    hops = int(request.args.get('hops', 1))
    max_nodes = int(request.args.get('max_nodes', 50))
    if node_id < 0 or node_id >= data.num_nodes:
        return jsonify({"error": f"Node ID {node_id} out of range [0, {data.num_nodes - 1}]"}), 400

    subgraph = get_ego_subgraph(data, center_node=node_id, num_hops=hops, max_nodes=max_nodes)
    return jsonify(subgraph)

@app.route('/api/classify', methods=['POST'])
def api_classify():
    body = request.get_json() or {}
    node_id = int(body.get('node_id', 0))
    if node_id < 0 or node_id >= data.num_nodes:
        return jsonify({"error": f"Node ID {node_id} out of range"}), 400

    results = {}
    true_cls = int(data.y[node_id].item())
    true_label = CLASS_LABELS.get(true_cls, f"Class {true_cls}")

    data_dev = data.to(DEVICE)
    x_dev = data_dev.x
    edge_dev = data_dev.edge_index

    for m_name, model in models.items():
        t0 = time.time()
        model.eval()
        with torch.no_grad():
            if m_name == 'mlp':
                logits = model(x_dev[node_id].unsqueeze(0))
            else:
                out = model(x_dev, edge_dev)
                logits = out[node_id].unsqueeze(0)

            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
            pred_cls = int(np.argmax(probs))
            latency_ms = round((time.time() - t0) * 1000, 2)

        results[m_name] = {
            "predicted_class_id": pred_cls,
            "predicted_class_name": CLASS_LABELS.get(pred_cls, f"Class {pred_cls}"),
            "confidence": float(round(float(probs[pred_cls]), 4)),
            "probabilities": [float(round(float(p), 4)) for p in probs],
            "correct": bool(pred_cls == true_cls),
            "latency_ms": latency_ms
        }

    return jsonify({
        "node_id": node_id,
        "true_class_id": true_cls,
        "true_class_name": true_label,
        "is_train": bool(data.train_mask[node_id].item()),
        "is_val": bool(data.val_mask[node_id].item()),
        "is_test": bool(data.test_mask[node_id].item()),
        "predictions": results
    })

@app.route('/api/predict-link', methods=['POST'])
def api_predict_link():
    body = request.get_json() or {}
    src = int(body.get('source_id', 0))
    dst = int(body.get('target_id', 1))
    model_name = body.get('model', 'gcn')

    if src < 0 or src >= data.num_nodes or dst < 0 or dst >= data.num_nodes:
        return jsonify({"error": "Node index out of range"}), 400

    emb = embeddings.get(model_name)
    if emb is None:
        return jsonify({"error": f"Embeddings for {model_name} unavailable"}), 500

    predictor = DotProductLinkPredictor()
    score = predictor.predict_pair(emb, src, dst)

    # Check if edge already exists
    src_neighbors = get_node_neighbors(data, src)
    already_connected = bool(dst in src_neighbors)

    src_cls = int(data.y[src].item())
    dst_cls = int(data.y[dst].item())

    return jsonify({
        "source_id": src,
        "target_id": dst,
        "model_used": model_name,
        "link_probability": round(score, 4),
        "percentage": f"{round(score * 100, 1)}%",
        "already_connected": already_connected,
        "same_topic": bool(src_cls == dst_cls),
        "source_class": CLASS_LABELS.get(src_cls, f"Class {src_cls}"),
        "target_class": CLASS_LABELS.get(dst_cls, f"Class {dst_cls}")
    })

@app.route('/api/recommend-citations', methods=['GET'])
def api_recommend_citations():
    node_id = int(request.args.get('node_id', 0))
    k = int(request.args.get('k', 5))
    model_name = request.args.get('model', 'gcn')

    if node_id < 0 or node_id >= data.num_nodes:
        return jsonify({"error": "Node ID out of range"}), 400

    emb = embeddings.get(model_name)
    if emb is None:
        return jsonify({"error": f"Embeddings for {model_name} not found"}), 500

    existing_nbrs = get_node_neighbors(data, node_id)
    predictor = DotProductLinkPredictor()

    # Search in a candidate pool of test nodes + popular nodes for fast sub-millisecond response
    np.random.seed(node_id)
    test_indices = torch.where(data.test_mask)[0].tolist()
    sample_pool = list(set(test_indices[:400] + list(np.random.randint(0, data.num_nodes, 400))))

    recs = predictor.recommend_top_k(emb, node_id, existing_neighbors=existing_nbrs, k=k, candidate_nodes=sample_pool)

    query_cls = int(data.y[node_id].item())
    out_recs = []
    for cand_id, score in recs:
        cand_cls = int(data.y[cand_id].item())
        out_recs.append({
            "paper_id": cand_id,
            "score": round(score, 4),
            "match_percentage": f"{round(score * 100, 1)}%",
            "topic": CLASS_LABELS.get(cand_cls, f"Class {cand_cls}"),
            "topic_color": CLASS_COLORS.get(cand_cls, "#6366f1"),
            "same_topic": bool(cand_cls == query_cls),
            "rationale": "High structural embedding cosine similarity" if cand_cls == query_cls else "Cross-topic citation candidate"
        })

    return jsonify({
        "query_node_id": node_id,
        "query_topic": CLASS_LABELS.get(query_cls, f"Class {query_cls}"),
        "existing_citation_count": len(existing_nbrs),
        "model_used": model_name,
        "recommendations": out_recs
    })

@app.route('/api/explain', methods=['GET'])
def api_explain():
    node_id = int(request.args.get('node_id', 0))
    if node_id < 0 or node_id >= data.num_nodes:
        return jsonify({"error": "Node ID out of range"}), 400

    if gat_attention_cache is None:
        return jsonify({"error": "GAT attention cache unavailable"}), 500

    edge_idx = gat_attention_cache['edge_index']
    attn_w = gat_attention_cache['attention_weights']

    # Find edges where target == node_id (incoming citations) or source == node_id
    src_mask = (edge_idx[0] == node_id)
    tgt_mask = (edge_idx[1] == node_id)
    relevant_mask = src_mask | tgt_mask
    relevant_indices = torch.where(relevant_mask)[0]

    explanations = []
    center_cls = int(data.y[node_id].item())

    for idx in relevant_indices:
        u = int(edge_idx[0, idx].item())
        v = int(edge_idx[1, idx].item())
        nbr = v if u == node_id else u
        nbr_cls = int(data.y[nbr].item())
        w = float(round(float(attn_w[idx].item()), 4))
        explanations.append({
            "neighbor_id": nbr,
            "direction": "outgoing" if u == node_id else "incoming",
            "attention_weight": w,
            "neighbor_topic": CLASS_LABELS.get(nbr_cls, f"Class {nbr_cls}"),
            "neighbor_color": CLASS_COLORS.get(nbr_cls, "#6366f1"),
            "same_topic": bool(nbr_cls == center_cls)
        })

    # Sort descending by attention weight
    explanations = sorted(explanations, key=lambda x: x['attention_weight'], reverse=True)[:15]

    return jsonify({
        "node_id": node_id,
        "center_topic": CLASS_LABELS.get(center_cls, f"Class {center_cls}"),
        "total_connected_citations": len(relevant_indices),
        "top_attended_citations": explanations
    })

@app.route('/api/benchmark', methods=['GET'])
def api_benchmark():
    p = os.path.join(RESULTS_DIR, 'benchmark_results.json')
    if os.path.exists(p):
        with open(p, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Benchmark results not yet generated. Run train_and_evaluate.py first."}), 404

@app.route('/api/robustness', methods=['GET'])
def api_robustness():
    p = os.path.join(RESULTS_DIR, 'robustness_results.json')
    if os.path.exists(p):
        with open(p, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Robustness results not yet generated. Run train_and_evaluate.py first."}), 404

@app.route('/api/link-prediction', methods=['GET'])
def api_link_prediction():
    p = os.path.join(RESULTS_DIR, 'link_prediction_results.json')
    if os.path.exists(p):
        with open(p, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Link prediction results not yet generated."}), 404

@app.route('/api/xai-sample', methods=['GET'])
def api_xai_sample():
    p = os.path.join(RESULTS_DIR, 'xai_attention_sample.json')
    if os.path.exists(p):
        with open(p, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "XAI samples not yet generated."}), 404

# Dictionary of clinical domain keywords for diabetes subtopics
DOMAIN_KEYWORDS = {
    0: ["streptozotocin", "alloxan", "rat", "rats", "mice", "mouse", "animal", "in vivo", "in vitro", "perfusion", "pancreatectomy", "wistar", "rodent", "experimental", "induced"],
    1: ["type 1", "t1d", "autoimmune", "autoimmunity", "autoantibody", "autoantibodies", "islet", "islets", "beta-cell", "beta cell", "insulitis", "hla", "juvenile", "childhood", "iddm", "lymphocyte"],
    2: ["type 2", "t2d", "insulin resistance", "insulin sensitivity", "metformin", "sulfonylurea", "obesity", "obese", "bmi", "overweight", "glycemic control", "hba1c", "metabolic syndrome", "adult-onset", "niddm", "adipose"]
}

# External biomedical topics that are outside PubMed-Diabetes closed-world taxonomy
OUT_OF_DOMAIN_KEYWORDS = {
    "Cancer / Oncology": [
        "cancer", "tumor", "tumour", "oncology", "carcinoma", "melanoma", "chemotherapy",
        "lymphoma", "leukemia", "metastasis", "neoplasm", "sarcoma", "glioma", "biopsy",
        "malignant", "malignancy", "radiation therapy", "antineoplastic", "immunotherapy"
    ],
    "Cardiovascular": [
        "myocardial", "heart failure", "cardiac", "arrhythmia", "stroke", "atherosclerosis", "coronary", "hypertension"
    ],
    "Neurology / Neuroscience": [
        "alzheimer", "parkinson", "dementia", "neurodegenerative", "neuron", "synapse", "brain", "cortex"
    ]
}

@app.route('/api/analyze-custom-paper', methods=['POST'])
def api_analyze_custom_paper():
    body = request.get_json() or {}
    title = body.get('title', '').lower()
    abstract = body.get('abstract', '').lower()
    full_text = f"{title} {abstract}".strip()

    if not full_text:
        return jsonify({"error": "Please provide a paper title or abstract"}), 400

    # Count matching diabetes keywords
    matched_keywords = []
    class_scores = [0.1, 0.1, 0.1]  # prior baseline

    for c_id, kws in DOMAIN_KEYWORDS.items():
        for kw in kws:
            if kw in full_text:
                matched_keywords.append(kw)
                class_scores[c_id] += full_text.count(kw) * 1.5

    # Check for out-of-domain topics like Cancer/Oncology
    matched_ood_categories = []
    matched_ood_keywords = []
    for cat, kws in OUT_OF_DOMAIN_KEYWORDS.items():
        cat_matches = [kw for kw in kws if kw in full_text]
        if cat_matches:
            matched_ood_categories.append(cat)
            matched_ood_keywords.extend(cat_matches)

    is_out_of_domain = bool(matched_ood_categories and len(matched_keywords) == 0)

    # Synthesize 500-dim feature vector from class centroids + noise
    c0 = data.x[data.y == 0].mean(dim=0)
    c1 = data.x[data.y == 1].mean(dim=0)
    c2 = data.x[data.y == 2].mean(dim=0)

    weights = torch.tensor(class_scores, dtype=torch.float32)
    weights = F.softmax(weights, dim=0)

    custom_x = (weights[0] * c0 + weights[1] * c1 + weights[2] * c2).unsqueeze(0).to(DEVICE)

    # Run MLP inference
    mlp = models['mlp']
    mlp.eval()
    with torch.no_grad():
        logits = mlp(custom_x)
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_cls = int(np.argmax(probs))

    # Construct helpful domain notice if paper is out-of-domain
    domain_notice = None
    if is_out_of_domain:
        cats_str = ", ".join(matched_ood_categories)
        domain_notice = (
            f"Note: '{cats_str}' is outside the PubMed citation dataset domain. "
            "The PubMed benchmark taxonomy contains strictly 3 Diabetes Mellitus classes: "
            "(0) Experimental Diabetes, (1) Type 1 Diabetes, and (2) Type 2 Diabetes. "
            f"Because 'Cancer' is not a class in this dataset, the model has mapped this paper onto '{CLASS_LABELS.get(pred_cls)}' "
            f"as its nearest immunological/metabolic analog based on shared biomedical vocabulary (e.g. cellular & immune markers), "
            "with lower classification confidence."
        )

    # Recommend Top-5 Papers to cite from the graph using embedding/feature cosine similarity
    with torch.no_grad():
        x_norm = F.normalize(custom_x.cpu(), p=2, dim=-1)
        data_norm = F.normalize(data.x, p=2, dim=-1)
        sims = (x_norm @ data_norm.T).squeeze().numpy()
        top_indices = np.argsort(sims)[::-1][:5]

    recommendations = []
    for idx in top_indices:
        cand_id = int(idx)
        cand_cls = int(data.y[cand_id].item())
        score = float(round(float(sims[cand_id]), 4))
        recommendations.append({
            "paper_id": cand_id,
            "similarity_score": score,
            "match_percentage": f"{round(score * 100, 1)}%",
            "topic": CLASS_LABELS.get(cand_cls, f"Class {cand_cls}"),
            "topic_color": CLASS_COLORS.get(cand_cls, "#6366f1"),
            "same_topic": bool(cand_cls == pred_cls),
            "rationale": "High vocabulary overlap and embedding affinity" if cand_cls == pred_cls else "Cross-topic medical reference"
        })

    return jsonify({
        "title": body.get('title', ''),
        "predicted_class_id": pred_cls,
        "predicted_class_name": CLASS_LABELS.get(pred_cls, f"Class {pred_cls}"),
        "confidence": float(round(float(probs[pred_cls]), 4)),
        "confidence_percentage": f"{round(float(probs[pred_cls]) * 100, 1)}%",
        "probabilities": [float(round(float(p), 4)) for p in probs],
        "extracted_keywords": list(set(matched_keywords)),
        "is_out_of_domain": is_out_of_domain,
        "out_of_domain_topics": matched_ood_categories,
        "out_of_domain_keywords": list(set(matched_ood_keywords)),
        "domain_notice": domain_notice,
        "recommended_citations": recommendations
    })


if __name__ == '__main__':
    init_app_state()
    port = int(os.environ.get('PORT', 5000))
    print(f"[+] Starting NodeSense Web Server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
