// NodeSense REST API Client
const API_BASE = '';

const API = {
  async getStatus() {
    const res = await fetch(`${API_BASE}/api/status`);
    return await res.json();
  },

  async getMetadata() {
    const res = await fetch(`${API_BASE}/api/metadata`);
    return await res.json();
  },

  async getSubgraph(nodeId = 0, hops = 1, maxNodes = 50) {
    const res = await fetch(`${API_BASE}/api/subgraph?node_id=${nodeId}&hops=${hops}&max_nodes=${maxNodes}`);
    return await res.json();
  },

  async classifyNode(nodeId) {
    const res = await fetch(`${API_BASE}/api/classify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ node_id: parseInt(nodeId) })
    });
    return await res.json();
  },

  async predictLink(sourceId, targetId, model = 'gcn') {
    const res = await fetch(`${API_BASE}/api/predict-link`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_id: parseInt(sourceId), target_id: parseInt(targetId), model })
    });
    return await res.json();
  },

  async recommendCitations(nodeId, k = 5, model = 'gcn') {
    const res = await fetch(`${API_BASE}/api/recommend-citations?node_id=${nodeId}&k=${k}&model=${model}`);
    return await res.json();
  },

  async getExplanation(nodeId) {
    const res = await fetch(`${API_BASE}/api/explain?node_id=${nodeId}`);
    return await res.json();
  },

  async getBenchmark() {
    const res = await fetch(`${API_BASE}/api/benchmark`);
    return await res.json();
  },

  async getRobustness() {
    const res = await fetch(`${API_BASE}/api/robustness`);
    return await res.json();
  },

  async getLinkPrediction() {
    const res = await fetch(`${API_BASE}/api/link-prediction`);
    return await res.json();
  },

  async getXAISample() {
    const res = await fetch(`${API_BASE}/api/xai-sample`);
    return await res.json();
  },

  async analyzeCustomPaper(title, abstract) {
    const res = await fetch(`${API_BASE}/api/analyze-custom-paper`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, abstract })
    });
    return await res.json();
  }
};

