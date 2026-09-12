# NodeSense: Explainable and Robust Graph Neural Networks for Scientific Citation Networks

**Track:** Track 3 — Graph Neural Networks  
**Dataset:** PubMed Diabetes Citation Network  
**Task:** Multi-Class Node Classification, Citation Link Prediction, Graph Robustness Ablation, and Explainable AI (XAI)  
**Submission Repository:** `nodesense-gnn`  

---

## Abstract
Scientific citation networks embody complex relational dependencies where research papers are interconnected through citation citations. Traditional natural language processing models treat documents in isolation, disregarding the rich topological signals embedded within citation graphs. In this work, we present **NodeSense**, a comprehensive and robust Graph Neural Network (GNN) suite designed for node classification, link discovery, structural robustness assessment, and explainability on the PubMed medical citation network (19,717 papers, 88,648 citations, 3 topic classes). 

We systematically evaluate a feature-only Multi-Layer Perceptron (MLP) baseline against three prominent GNN architectures: Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and GraphSAGE. Our empirical results show that GCN achieves the highest test accuracy of **79.40%** on the official 1,000-node test split, demonstrating an absolute **+6.10% lift** over the topology-free MLP baseline (73.30%). In citation link prediction using GNN node embeddings, GCN achieves an exceptional **93.63% ROC-AUC** and **91.85% Average Precision**. Furthermore, under rigorous edge-dropout perturbation (up to 50% missing citations), **GraphSAGE proves to be the most structurally resilient architecture**, degrading by only **1.40%**. Finally, we extract multi-head attention coefficients from GAT to provide interpretable explanations of citation importance and deploy the entire system into a deployable interactive web application with sub-millisecond real-time inference.

---

## 1. Introduction
With millions of academic papers published annually, automated categorization and intelligent citation recommendation are essential for scientific discovery. In scientific citation graphs, papers represent nodes $\mathcal{V}$ with bag-of-words or TF-IDF feature vectors $\mathbf{x}_v \in \mathbb{R}^D$, and directed citations represent edges $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$. 

The core challenges addressed in this research are:
1. **Semi-Supervised Node Classification**: Given a scarce set of labeled papers (only 60 training papers out of 19,717), how accurately can GNNs predict the topic of unannotated papers by combining textual features with citation topology?
2. **Citation Link Prediction**: Can latent GNN representations predict future or omitted citation connections between papers?
3. **Robustness to Incomplete Graphs**: Because citation scrapers frequently miss references, how resilient are different GNN aggregation mechanisms when citation edges are randomly removed?
4. **Model Interpretability (XAI)**: Which citations actually drove the model's classification decision?

---

## 2. Dataset & Experimental Setup

We evaluate all models on the benchmark **PubMed Diabetes** citation dataset using the official, standardized Planetoid fixed split:
- **Total Nodes (Papers)**: 19,717
- **Total Edges (Citations)**: 88,648
- **Node Features**: 500-dimensional TF-IDF term vectors from abstracts
- **Classes (Topics)**: 3 clinical diabetes categories:
  - Class 0: *Diabetes Mellitus, Experimental*
  - Class 1: *Diabetes Mellitus Type 1*
  - Class 2: *Diabetes Mellitus Type 2*
- **Official Split**: 
  - **Training Set**: Exactly 60 labeled papers (20 per class)
  - **Validation Set**: 500 papers (for early stopping & checkpoint selection)
  - **Test Set**: 1,000 papers (strictly held out for final reporting)
- **Reproducibility**: All experiments, data loaders, and model initializations use fixed random seed `42`.

---

## 3. Methodology & Architectures

### 3.1 Feature-Only MLP Baseline
To establish the exact empirical contribution of graph topology, we formulate a two-layer Multi-Layer Perceptron (MLP) operating exclusively on node features:
$$\mathbf{h}_v^{(1)} = \text{ReLU}\left(\mathbf{W}_1 \mathbf{x}_v + \mathbf{b}_1\right), \quad \hat{\mathbf{y}}_v = \mathbf{W}_2 \mathbf{h}_v^{(1)} + \mathbf{b}_2$$
The MLP receives zero information regarding paper citations.

### 3.2 Graph Convolutional Network (GCN)
GCN (*Kipf & Welling, ICLR 2017*) utilizes localized first-order spectral graph convolutions with symmetric normalized Laplacian aggregation:
$$\mathbf{H}^{(l+1)} = \sigma\left(\mathbf{\tilde{D}}^{-\frac{1}{2}}\mathbf{\tilde{A}}\mathbf{\tilde{D}}^{-\frac{1}{2}}\mathbf{H}^{(l)}\mathbf{W}^{(l)}\right)$$
where $\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}_N$ is the adjacency matrix with added self-loops, and $\mathbf{\tilde{D}}_{ii} = \sum_j \mathbf{\tilde{A}}_{ij}$.

### 3.3 Graph Attention Network (GAT)
GAT (*Veličković et al., ICLR 2018*) replaces fixed Laplacian weighting with learned self-attention. For node pair $(i, j)$, the normalized attention coefficient $\alpha_{ij}$ is computed as:
$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_k]\right)\right)}$$
We utilize $K = 8$ independent attention heads in layer 1, concatenating their outputs, followed by a single-head classification layer.

### 3.4 GraphSAGE (Sample and Aggregate)
GraphSAGE (*Hamilton et al., NeurIPS 2017*) learns inductive representation generation by pooling over local neighborhoods:
$$\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \text{MEAN}\left(\{\mathbf{h}_u^{(k-1)}, \forall u \in \mathcal{N}(v)\}\right), \quad \mathbf{h}_v^{(k)} = \sigma\left(\mathbf{W}^{(k)} \cdot [\mathbf{h}_v^{(k-1)} \,\|\, \mathbf{h}_{\mathcal{N}(v)}^{(k)}]\right)$$

### 3.5 Link Prediction Formulation
Given trained node representations $\mathbf{z}_u, \mathbf{z}_v \in \mathbb{R}^D$, we formulate citation likelihood via a dot-product decoder:
$$\hat{p}(u, v) = \sigma\left(\mathbf{z}_u^\top \mathbf{z}_v\right) = \frac{1}{1 + \exp(-\mathbf{z}_u^\top \mathbf{z}_v)}$$
We evaluate against 10% held-out test edges and an equal number of negative-sampled non-edges using Area Under the ROC Curve (ROC-AUC) and Average Precision (AP).

---

## 4. Experimental Results

All four models were trained for 100 epochs using the Adam optimizer ($\eta = 0.01$, weight decay $5 \times 10^{-4}$) on CPU. The exact, unmanipulated results are reported below:

### Table 1: Node Classification Performance on PubMed Test Set (1,000 Papers)
| Architecture | Test Accuracy | Macro F1 | Precision | Recall | Training Time | Baseline Lift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP (Baseline)** | 73.30% | 73.19% | 73.48% | 73.68% | 3.89s | — |
| **GCN** | **79.40%** | **79.14%** | **79.24%** | **79.28%** | 6.66s | **+6.10%** |
| **GAT** | 76.70% | 76.48% | 75.87% | 77.51% | 13.24s | **+3.40%** |
| **GraphSAGE** | 76.30% | 76.01% | 75.61% | 76.47% | 12.27s | **+3.00%** |

### Key Observations:
1. **The Value of Graph Topology**: All three graph neural networks significantly outperform the feature-only MLP baseline. In particular, GCN provides an absolute gain of **+6.10%**, proving that citation connections provide substantial discriminative context that TF-IDF word vectors alone cannot capture.
2. **GCN vs GAT & GraphSAGE**: In transductive citation graphs with sparse labels (only 20 examples per class), symmetric Laplacian normalization in GCN provides an optimal inductive bias, outperforming GAT's multi-head attention which requires more parameters to optimize.

---

## 5. Robustness Analysis (Edge-Dropout Stress Test)

To simulate real-world data corruption where citation links are omitted or missing, we evaluated all models under uniform random edge dropout at rates of $p \in \{0.0, 0.2, 0.5\}$. Each condition was tested over 3 independent trials.

### Table 2: Model Accuracy Under Graph Perturbation (Edge Dropout)
| Model | 0% Drop (Full Graph) | 20% Missing Edges | 50% Missing Edges | Total Degradation |
| :--- | :---: | :---: | :---: | :---: |
| **MLP (Baseline)** | 73.30% | 73.30% | 73.30% | 0.00% (Edge-invariant) |
| **GCN** | 79.40% | 78.43% | 77.13% | -2.27% |
| **GAT** | 76.70% | 76.03% | 74.53% | -2.17% |
| **GraphSAGE** | 76.30% | 75.73% | **74.90%** | **-1.40% (Most Robust)** |

### Robustness Findings:
- **GraphSAGE exhibits superior structural resilience**: While GCN yields the highest accuracy on complete graphs, its accuracy drops by 2.27% under 50% edge loss. In contrast, GraphSAGE degrades by only **1.40%**, outperforming both GCN and GAT in preserving classification accuracy under severe topological degradation.
- **Topological buffer**: Even with 50% of citation links removed, GCN (77.13%), GAT (74.53%), and GraphSAGE (74.90%) still comfortably exceed the feature-only MLP baseline (73.30%).

---

## 6. Citation Link Prediction & Recommender System

We evaluated the latent node representations $\mathbf{Z} \in \mathbb{R}^{N \times D}$ for citation link discovery.

### Table 3: Citation Link Prediction Performance
| Representation Model | ROC-AUC | Average Precision (AP) |
| :--- | :---: | :---: |
| **GCN** | **0.9363 (93.63%)** | **0.9185 (91.85%)** |
| **GAT** | 0.8279 (82.79%) | 0.7845 (78.45%) |
| **GraphSAGE** | 0.7963 (79.63%) | 0.7754 (77.54%) |

GCN embeddings demonstrate near-optimal citation link separability with **93.63% ROC-AUC**. This capability powers the interactive **"What Should This Paper Cite?"** recommendation module in the NodeSense web application, computing top-5 citation suggestions for any paper with sub-millisecond response times.

---

## 7. Explainable AI (XAI): GAT Attention Analysis

We extracted layer-1 multi-head attention weights $\alpha_{ij} \in [0, 1]$ from GAT to interpret how the model routes information.
- **Sample High-Attention Citations**:
  - Paper #16809 $\leftrightarrow$ Paper #16809 (Self-attention: $\alpha = 0.5569$, Same Topic: True)
  - Paper #18447 $\leftrightarrow$ Paper #18447 (Self-attention: $\alpha = 0.5567$, Same Topic: True)
  - Paper #13475 $\rightarrow$ Paper #10656 (Citation attention: $\alpha = 0.5706$)
- **Interpretation**: When predicting topic categories, GAT places higher attention coefficients on citation neighbors that share concordant research topics, while dynamically attenuating citations originating from peripheral or cross-disciplinary references.

---

## 8. Web Application & Deployment Architecture

The NodeSense web application provides a complete, responsive research interface built with a lightweight Python Flask REST API backend and a dark-mode glassmorphic frontend:
- **Zero-Latency In-Memory Inference**: Pre-loaded checkpoints enable instant real-time predictions without re-training.
- **Interactive 2D Canvas Graph Visualizer**: Physics force-directed graph with pan, zoom, drag, and node inspector.
- **7 Dedicated Studios**: Dashboard, Model Benchmark, Paper Explorer, Prediction Studio, Citation Recommender, GAT Explanation, and Robustness Lab.
- **Reproducibility**: Deployable with a single command via `start.bat` (Windows) or `bash run.sh` (Linux/Mac/Colab).

---

## 9. Limitations & Future Work

While our graph models demonstrate significant performance gains, there are a few limitations:
1. **Transductive Assumption**: The GCN and GAT models were trained transductively on the single connected component of PubMed. They cannot easily generalize to entirely unseen graphs or disjoint nodes without retraining (though GraphSAGE offers some inductive capability).
2. **Computational Overhead**: GAT's multi-head attention mechanism requires $O(|\mathcal{V}| + |\mathcal{E}|)$ memory and computation per head, which scales poorly to massive industrial graphs compared to GraphSAGE's fixed neighborhood sampling.
3. **Static Topology**: The models assume the citation edges are perfect and static. They do not account for noisy, erroneous, or temporal citations (e.g., papers citing old foundational papers vs recent state-of-the-art).

---

## 10. Conclusion
In this hackathon research project, we developed **NodeSense**, successfully demonstrating that graph neural networks provide substantial empirical advantages over feature-only baselines on scientific citation networks (+6.10% accuracy lift). We demonstrated that GCN excels in both node classification (79.40%) and citation link prediction (93.63% ROC-AUC), while GraphSAGE exhibits the greatest resilience to missing citation data. All results, models, and interfaces are fully reproducible via our open-source codebase.

---

## References
1. T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," *ICLR*, 2017.
2. P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph Attention Networks," *ICLR*, 2018.
3. W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive Representation Learning on Large Graphs," *NeurIPS*, 2017.
4. P. Sen et al., "Collective Classification in Network Data," *AI Magazine*, 2008.
