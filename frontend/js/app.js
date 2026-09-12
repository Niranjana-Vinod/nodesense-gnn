// NodeSense Application Controller
document.addEventListener('DOMContentLoaded', async () => {
  let visualizer = null;
  let currentSelectedNode = 100; // default node to explore
  let cachedRobustness = null; // cached robustness ablation results

  // Tab switching
  const tabs = document.querySelectorAll('.nav-tab');
  const panes = document.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.dataset.tab;
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');

      if (targetId === 'explorer-tab' && visualizer) {
        visualizer.resize();
      }
    });
  });

  // Initialize Canvas Visualizer
  visualizer = new GraphVisualizer('graphCanvas', (selectedId) => {
    selectNode(selectedId);
  });

  document.getElementById('btnResetZoom')?.addEventListener('click', () => {
    visualizer.resetView();
  });

  // Search input in Explorer
  document.getElementById('btnExplore')?.addEventListener('click', () => {
    const rawVal = (document.getElementById('inputNodeSearch').value || '').trim();
    const val = parseInt(rawVal);
    if (!isNaN(val)) {
      if (val >= 0 && val < 19717) {
        selectNode(val);
      } else {
        alert('Paper ID must be between 0 and 19,716 (total PubMed nodes).');
      }
    } else {
      alert(`"${rawVal}" is not a numeric Paper ID. Paper Explorer navigates by Paper ID (0–19716). To analyze paper text and keywords like "${rawVal}", please switch to the "Analyze Custom Paper" tab!`);
    }
  });

  // Quick preset node buttons
  document.querySelectorAll('.btn-sample-node').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const nid = parseInt(e.target.dataset.node);
      selectNode(nid);
    });
  });

  // Classify Button
  document.getElementById('btnRunClassify')?.addEventListener('click', () => {
    const val = parseInt(document.getElementById('inputClassifyNode').value);
    if (!isNaN(val)) {
      runClassification(val);
    }
  });

  // Recommend Citations Button
  document.getElementById('btnRunRecommend')?.addEventListener('click', () => {
    const val = parseInt(document.getElementById('inputRecommendNode').value);
    const model = document.getElementById('selectRecommendModel').value;
    if (!isNaN(val)) {
      runCitationRecommendation(val, model);
    }
  });

  // Link Pair Predictor Button
  document.getElementById('btnPredictPair')?.addEventListener('click', () => {
    const src = parseInt(document.getElementById('inputPairSrc').value);
    const dst = parseInt(document.getElementById('inputPairDst').value);
    const model = document.getElementById('selectPairModel').value;
    if (!isNaN(src) && !isNaN(dst)) {
      runPairPrediction(src, dst, model);
    }
  });

  // Explain Button
  document.getElementById('btnRunExplain')?.addEventListener('click', () => {
    const val = parseInt(document.getElementById('inputExplainNode').value);
    if (!isNaN(val)) {
      runExplanation(val);
    }
  });

  // Robustness Slider
  const robSlider = document.getElementById('robustnessSlider');
  const robSliderVal = document.getElementById('robustnessSliderVal');
  if (robSlider && robSliderVal) {
    robSlider.addEventListener('input', (e) => {
      const drop = e.target.value;
      robSliderVal.textContent = `${drop}%`;
      updateRobustnessView(drop);
    });
  }

  // Custom Paper Presets
  const sampleAbstracts = {
    "1": {
      title: "Autoantibodies against islet antigens and beta-cell autoimmunity in juvenile diabetes",
      abstract: "We investigated circulating autoantibodies targeting GAD65 and IA-2 antigens in childhood patients with newly diagnosed type 1 diabetes mellitus. Insulitis and progressive destruction of pancreatic beta cells were correlated with HLA genetic susceptibility."
    },
    "2": {
      title: "Metformin efficacy, insulin resistance, and glycemic control in obese adult patients",
      abstract: "A clinical trial evaluating metformin and sulfonylurea therapy on insulin sensitivity and HbA1c reduction in adult-onset type 2 diabetes. Body mass index, metabolic syndrome, and cardiovascular risk factors showed substantial improvement."
    },
    "0": {
      title: "Streptozotocin-induced diabetic nephropathy and pancreatic damage in Wistar rat models",
      abstract: "In vivo laboratory investigation using streptozotocin injection to induce experimental diabetes mellitus in male Wistar rats. Pancreatic perfusion and histological tissue examination revealed severe beta-cell necrosis."
    },
    "cancer": {
      title: "Targeted immunotherapy and checkpoint inhibition in metastatic lung cancer and melanoma",
      abstract: "Investigation of PD-1/PD-L1 inhibitors and cytotoxic T-cell activation in non-small cell lung carcinoma. Observed tumor microenvironment modulation, lymphocyte infiltration, and systemic immune-related adverse events."
    }
  };

  document.querySelectorAll('.btn-sample-abstract').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const type = e.target.dataset.type;
      const data = sampleAbstracts[type];
      if (data) {
        document.getElementById('inputCustomTitle').value = data.title;
        document.getElementById('inputCustomAbstract').value = data.abstract;
      }
    });
  });

  // Analyze Custom Paper Button
  document.getElementById('btnAnalyzeCustomPaper')?.addEventListener('click', async () => {
    const title = document.getElementById('inputCustomTitle').value.trim();
    const abstract = document.getElementById('inputCustomAbstract').value.trim();
    const container = document.getElementById('customPaperResultContainer');

    if (!title && !abstract) {
      alert('Please enter a paper title or abstract, or select a quick preset.');
      return;
    }

    container.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 30px;">Analyzing paper semantics & searching citation graph...</div>`;

    try {
      const res = await API.analyzeCustomPaper(title, abstract);
      const colorMap = { 0: '#3b82f6', 1: '#10b981', 2: '#f59e0b' };
      const color = colorMap[res.predicted_class_id] || '#818cf8';

      let recsHtml = '';
      res.recommended_citations.forEach((r, idx) => {
        recsHtml += `
          <div class="rec-card">
            <div>
              <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-weight: 700; color: #818cf8;">#${idx + 1}</span>
                <strong>PubMed Paper #${r.paper_id}</strong>
                <span class="badge" style="background: rgba(255,255,255,0.06); color: ${r.topic_color}; border: 1px solid ${r.topic_color}40;">${r.topic}</span>
              </div>
              <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
                ${r.rationale} &bull; ${r.same_topic ? 'Direct Topic Overlap' : 'Interdisciplinary Context'}
              </div>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 18px; font-weight: 700; color: #10b981;">${r.match_percentage}</div>
              <div style="font-size: 10px; color: var(--text-muted);">Citation Affinity</div>
            </div>
          </div>
        `;
      });

      let kwHtml = '';
      if (res.extracted_keywords && res.extracted_keywords.length > 0) {
        kwHtml = res.extracted_keywords.map(k => `<span class="badge badge-info" style="margin-right: 4px; margin-bottom: 4px;">${k}</span>`).join('');
      } else {
        kwHtml = `<span style="color: var(--text-muted); font-size: 12px;">General biomedical vocabulary</span>`;
      }

      let domainNoticeHtml = '';
      if (res.is_out_of_domain && res.domain_notice) {
        const oodBadges = (res.out_of_domain_keywords || []).map(k => `<span class="badge badge-warning" style="margin-right: 4px; margin-bottom: 4px;">${k}</span>`).join('');
        domainNoticeHtml = `
          <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: var(--radius-sm); padding: 14px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 8px; color: #f59e0b; font-weight: 700; margin-bottom: 6px; font-size: 13px;">
              <span>⚠️ Topic Scope Notice: External Domain Detected</span>
            </div>
            <div style="font-size: 12px; color: var(--text-secondary); line-height: 1.5; margin-bottom: 8px;">
              ${res.domain_notice}
            </div>
            <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 4px;">
              <span style="font-size: 11px; color: var(--text-muted); margin-right: 4px;">External Keywords Identified:</span>
              ${oodBadges}
            </div>
          </div>
        `;
      }

      container.innerHTML = `
        ${domainNoticeHtml}
        <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: var(--radius-sm); padding: 16px; margin-bottom: 16px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div class="stat-label">${res.is_out_of_domain ? 'Nearest Analogous Category' : 'Predicted Category'}</div>
            <span class="badge badge-success" style="font-size: 13px; padding: 4px 10px; background: ${color}20; color: ${color}; border: 1px solid ${color}40;">
              ${res.predicted_class_name}
            </span>
          </div>
          <div style="font-size: 22px; font-weight: 700; color: ${color};">
            Confidence: ${res.confidence_percentage}
          </div>
          <div style="margin-top: 10px;">
            <div class="stat-label" style="margin-bottom: 4px;">Detected Domain Keywords:</div>
            <div style="display: flex; flex-wrap: wrap;">${kwHtml}</div>
          </div>
        </div>

        <div class="stat-label" style="margin-bottom: 10px;">Top 5 Papers in PubMed to Cite:</div>
        <div>${recsHtml}</div>
      `;
    } catch (e) {
      container.innerHTML = `<div style="color: var(--accent-danger); padding: 20px;">Error analyzing paper: ${e.message}</div>`;
    }
  });


  // Load initial data
  await loadSystemStatus();
  await loadMetadata();
  await loadBenchmark();
  await loadRobustness();
  await loadLinkPredictionBenchmark();
  await loadXAISamples();

  // Load default node
  selectNode(currentSelectedNode);

  // --- Functions ---
  async function loadSystemStatus() {
    try {
      const status = await API.getStatus();
      document.getElementById('deviceStatus').textContent = `${status.device.toUpperCase()}`;
      document.getElementById('modelsStatus').textContent = `${status.models_loaded.length} Loaded`;
    } catch (e) {
      console.warn('Backend connecting...', e);
    }
  }

  async function loadMetadata() {
    try {
      const meta = await API.getMetadata();
      document.getElementById('statNodes').textContent = Number(meta.num_nodes).toLocaleString();
      document.getElementById('statEdges').textContent = Number(meta.num_edges).toLocaleString();
      document.getElementById('statFeatures').textContent = meta.num_features;
      document.getElementById('statClasses').textContent = meta.num_classes;

      // Class breakdown
      const classContainer = document.getElementById('classDistributionList');
      if (classContainer && meta.class_distribution) {
        classContainer.innerHTML = '';
        for (const [name, count] of Object.entries(meta.class_distribution)) {
          const div = document.createElement('div');
          div.className = 'rec-card';
          div.innerHTML = `
            <div>
              <strong>${name}</strong>
              <div style="font-size: 11px; color: var(--text-secondary);">PubMed Citation Topic</div>
            </div>
            <div style="font-size: 16px; font-weight: 700;">${count.toLocaleString()} papers</div>
          `;
          classContainer.appendChild(div);
        }
      }
    } catch (e) {
      console.error('Error loading metadata:', e);
    }
  }

  async function loadBenchmark() {
    try {
      const data = await API.getBenchmark();
      const tbody = document.getElementById('benchmarkTableBody');
      if (!tbody) return;
      tbody.innerHTML = '';

      let bestAcc = 0;
      let bestModel = '';

      data.forEach(item => {
        if (item.accuracy > bestAcc) {
          bestAcc = item.accuracy;
          bestModel = item.model;
        }

        const tr = document.createElement('tr');
        const badge = item.beats_baseline === true 
          ? `<span class="badge badge-success">+${(item.delta_vs_baseline * 100).toFixed(2)}% vs MLP</span>`
          : item.beats_baseline === false
            ? `<span class="badge badge-warning">Below MLP</span>`
            : `<span class="badge badge-info">Baseline</span>`;

        tr.innerHTML = `
          <td><strong>${item.model}</strong></td>
          <td><strong style="color: #818cf8;">${(item.accuracy * 100).toFixed(2)}%</strong></td>
          <td>${(item.macro_f1 * 100).toFixed(2)}%</td>
          <td>${(item.precision * 100).toFixed(2)}%</td>
          <td>${(item.recall * 100).toFixed(2)}%</td>
          <td>${item.training_time_sec}s</td>
          <td>${badge}</td>
        `;
        tbody.appendChild(tr);
      });

      if (document.getElementById('statBestModel')) {
        document.getElementById('statBestModel').textContent = `${bestModel} (${(bestAcc * 100).toFixed(1)}%)`;
      }
    } catch (e) {
      console.error('Error loading benchmark:', e);
    }
  }

  async function loadRobustness() {
    try {
      cachedRobustness = await API.getRobustness();
      updateRobustnessView(0);
    } catch (e) {
      console.error('Error loading robustness:', e);
    }
  }

  function updateRobustnessView(dropVal) {
    if (!cachedRobustness) return;
    const dropPct = parseFloat(dropVal);
    const tbody = document.getElementById('robustnessTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    for (const [modelName, drops] of Object.entries(cachedRobustness)) {
      const baseAcc = drops['0.0'];
      let currentAcc;
      if (dropPct <= 0) {
        currentAcc = drops['0.0'];
      } else if (dropPct <= 20) {
        const ratio = dropPct / 20.0;
        currentAcc = drops['0.0'] + ratio * (drops['0.2'] - drops['0.0']);
      } else {
        const ratio = (dropPct - 20.0) / 30.0;
        currentAcc = drops['0.2'] + ratio * (drops['0.5'] - drops['0.2']);
      }
      const dropDelta = currentAcc - baseAcc;

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${modelName}</strong></td>
        <td>${(drops['0.0'] * 100).toFixed(2)}%</td>
        <td>${(drops['0.2'] * 100).toFixed(2)}%</td>
        <td>${(drops['0.5'] * 100).toFixed(2)}%</td>
        <td><strong style="color: ${dropDelta < -0.02 ? '#ef4444' : '#10b981'};">${(currentAcc * 100).toFixed(2)}%</strong></td>
        <td><span class="badge ${dropDelta < -0.02 ? 'badge-warning' : 'badge-success'}">${(dropDelta * 100).toFixed(2)}% drop</span></td>
      `;
      tbody.appendChild(tr);
    }
  }

  async function loadLinkPredictionBenchmark() {
    try {
      const data = await API.getLinkPrediction();
      const container = document.getElementById('linkPredictionResults');
      if (!container) return;
      container.innerHTML = '';

      for (const [mName, metrics] of Object.entries(data)) {
        const div = document.createElement('div');
        div.className = 'stat-box';
        div.innerHTML = `
          <div class="stat-label">${mName} Link Predictor</div>
          <div class="stat-value" style="color: #10b981;">ROC-AUC: ${(metrics.roc_auc * 100).toFixed(2)}%</div>
          <div class="stat-sub" style="color: var(--text-secondary);">Average Precision: ${(metrics.average_precision * 100).toFixed(2)}%</div>
        `;
        container.appendChild(div);
      }
    } catch (e) {
      console.error('Error loading link prediction benchmark:', e);
    }
  }

  async function loadXAISamples() {
    try {
      const samples = await API.getXAISample();
      const container = document.getElementById('xaiSampleList');
      if (!container || !samples || !samples.length) return;
      container.innerHTML = '';

      samples.slice(0, 5).forEach(item => {
        const div = document.createElement('div');
        div.className = 'rec-card';
        div.innerHTML = `
          <div>
            <strong>Paper #${item.source_node} &rarr; Paper #${item.target_node}</strong>
            <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">
              ${item.source_class} &bull; ${item.same_topic ? '<span style="color: #10b981;">Same Topic</span>' : '<span style="color: #f59e0b;">Cross-Topic</span>'}
            </div>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 16px; font-weight: 700; color: #818cf8;">${(item.attention_weight * 100).toFixed(1)}%</div>
            <div style="font-size: 10px; color: var(--text-muted);">Attention Weight</div>
          </div>
        `;
        container.appendChild(div);
      });
    } catch (e) {
      console.error('Error loading XAI samples:', e);
    }
  }

  async function selectNode(nodeId) {
    currentSelectedNode = nodeId;
    // Update inputs
    if (document.getElementById('inputNodeSearch')) document.getElementById('inputNodeSearch').value = nodeId;
    if (document.getElementById('inputClassifyNode')) document.getElementById('inputClassifyNode').value = nodeId;
    if (document.getElementById('inputRecommendNode')) document.getElementById('inputRecommendNode').value = nodeId;
    if (document.getElementById('inputPairSrc')) document.getElementById('inputPairSrc').value = nodeId;
    if (document.getElementById('inputExplainNode')) document.getElementById('inputExplainNode').value = nodeId;

    // Load Subgraph into Canvas
    try {
      const subgraph = await API.getSubgraph(nodeId, 1, 45);
      visualizer.setData(subgraph);
      updateInspector(subgraph, nodeId);
    } catch (e) {
      console.error('Error loading subgraph:', e);
    }

    // Automatically trigger classification, recommendation, and explanation for this node
    runClassification(nodeId);
    runCitationRecommendation(nodeId, 'gcn');
    runExplanation(nodeId);
  }

  function updateInspector(subgraph, centerId) {
    const centerNode = subgraph.nodes.find(n => n.id === centerId) || subgraph.nodes[0];
    const neighborCount = subgraph.nodes.length - 1;

    document.getElementById('inspectNodeId').textContent = `Paper #${centerNode.id}`;
    document.getElementById('inspectTopic').textContent = centerNode.class_name;
    document.getElementById('inspectTopic').style.color = centerNode.color;
    document.getElementById('inspectDegree').textContent = `${neighborCount} visible citations`;

    const splitBadge = centerNode.is_train ? 'Train Set (60)' : centerNode.is_test ? 'Test Set (1000)' : 'Validation / Unlabeled';
    document.getElementById('inspectSplit').textContent = splitBadge;
  }

  async function runClassification(nodeId) {
    const container = document.getElementById('classifyResultsContainer');
    if (!container) return;
    container.innerHTML = `<div style="color: var(--text-muted); padding: 20px;">Computing real-time inference across 4 models...</div>`;

    try {
      const res = await API.classifyNode(nodeId);
      container.innerHTML = '';

      document.getElementById('classifyNodeHeader').textContent = `Paper #${res.node_id} (True: ${res.true_class_name})`;

      const modelNames = {
        'mlp': 'MLP (Baseline)',
        'gcn': 'GCN',
        'gat': 'GAT',
        'sage': 'GraphSAGE'
      };

      for (const [key, p] of Object.entries(res.predictions)) {
        const div = document.createElement('div');
        div.className = 'stat-box';
        const isCorrect = p.correct;

        div.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="stat-label">${modelNames[key] || key}</div>
            <span class="badge ${isCorrect ? 'badge-success' : 'badge-warning'}">${isCorrect ? 'Correct' : 'Mismatch'}</span>
          </div>
          <div style="font-size: 16px; font-weight: 700; margin-top: 6px; color: ${isCorrect ? '#10b981' : '#f59e0b'};">
            ${p.predicted_class_name}
          </div>
          <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">
            Confidence: ${(p.confidence * 100).toFixed(1)}% &bull; Latency: ${p.latency_ms}ms
          </div>
          <div class="prob-bar-container">
            <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--text-muted);">
              <span>Exp: ${(p.probabilities[0]*100).toFixed(0)}%</span>
              <span>T1D: ${(p.probabilities[1]*100).toFixed(0)}%</span>
              <span>T2D: ${(p.probabilities[2]*100).toFixed(0)}%</span>
            </div>
            <div class="prob-bar-track">
              <div class="prob-bar-fill" style="width: ${(p.confidence * 100).toFixed(1)}%; background: ${isCorrect ? 'var(--accent-primary)' : 'var(--accent-warning)'};"></div>
            </div>
          </div>
        `;
        container.appendChild(div);
      }
    } catch (e) {
      container.innerHTML = `<div style="color: var(--accent-danger);">Error: ${e.message}</div>`;
    }
  }

  async function runCitationRecommendation(nodeId, model = 'gcn') {
    const list = document.getElementById('recommendationsList');
    if (!list) return;
    list.innerHTML = `<div style="color: var(--text-muted); padding: 12px;">Searching candidate citation space using ${model.toUpperCase()} embeddings...</div>`;

    try {
      const res = await API.recommendCitations(nodeId, 5, model);
      list.innerHTML = '';

      if (!res.recommendations || res.recommendations.length === 0) {
        list.innerHTML = `<div style="color: var(--text-muted);">No candidate recommendations available.</div>`;
        return;
      }

      res.recommendations.forEach((rec, idx) => {
        const item = document.createElement('div');
        item.className = 'rec-card';
        item.innerHTML = `
          <div>
            <div style="display: flex; align-items: center; gap: 8px;">
              <span style="font-weight: 700; color: #818cf8;">#${idx + 1}</span>
              <strong>Paper #${rec.paper_id}</strong>
              <span class="badge" style="background: rgba(255,255,255,0.06); color: ${rec.topic_color}; border: 1px solid ${rec.topic_color}40;">${rec.topic}</span>
            </div>
            <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
              ${rec.rationale} &bull; ${rec.same_topic ? 'Identical Topic' : 'Interdisciplinary Citation'}
            </div>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 18px; font-weight: 700; color: #10b981;">${rec.match_percentage}</div>
            <div style="font-size: 10px; color: var(--text-muted);">Link Likelihood</div>
          </div>
        `;
        list.appendChild(item);
      });
    } catch (e) {
      list.innerHTML = `<div style="color: var(--accent-danger);">Error: ${e.message}</div>`;
    }
  }

  async function runPairPrediction(src, dst, model) {
    const resultBox = document.getElementById('pairPredictionResult');
    if (!resultBox) return;
    resultBox.innerHTML = `Predicting...`;

    try {
      const res = await API.predictLink(src, dst, model);
      resultBox.innerHTML = `
        <div style="font-size: 20px; font-weight: 700; color: #10b981; margin-bottom: 4px;">
          Citation Probability: ${res.percentage}
        </div>
        <div style="font-size: 13px; color: var(--text-secondary);">
          Paper #${res.source_id} (${res.source_class}) &harr; Paper #${res.target_id} (${res.target_class})<br>
          Connected in Ground Truth: <strong>${res.already_connected ? 'YES' : 'NO (Discovered Citation Candidate)'}</strong>
        </div>
      `;
    } catch (e) {
      resultBox.innerHTML = `<span style="color: var(--accent-danger);">Error: ${e.message}</span>`;
    }
  }

  async function runExplanation(nodeId) {
    const list = document.getElementById('explanationList');
    if (!list) return;
    list.innerHTML = `<div style="color: var(--text-muted); padding: 12px;">Extracting GAT multi-head attention coefficients...</div>`;

    try {
      const res = await API.getExplanation(nodeId);
      list.innerHTML = '';

      if (!res.top_attended_citations || res.top_attended_citations.length === 0) {
        list.innerHTML = `<div style="color: var(--text-muted); padding: 12px;">No incoming or outgoing citation edges found for Paper #${nodeId}.</div>`;
        return;
      }

      res.top_attended_citations.forEach((att, i) => {
        const item = document.createElement('div');
        item.className = 'rec-card';
        item.innerHTML = `
          <div>
            <div style="display: flex; align-items: center; gap: 8px;">
              <span style="font-weight: 700; color: var(--accent-primary);">#${i + 1}</span>
              <strong>Neighbor Paper #${att.neighbor_id}</strong>
              <span class="badge" style="background: rgba(255,255,255,0.06); color: ${att.neighbor_color};">${att.neighbor_topic}</span>
            </div>
            <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
              ${att.direction === 'outgoing' ? '&rarr; Outgoing Citation' : '&larr; Cited By Neighbor'} &bull; 
              ${att.same_topic ? '<span style="color: #10b981;">Topic Concordant</span>' : '<span style="color: #f59e0b;">Cross-Disciplinary</span>'}
            </div>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 18px; font-weight: 700; color: #818cf8;">${(att.attention_weight * 100).toFixed(1)}%</div>
            <div style="font-size: 10px; color: var(--text-muted);">GAT Attention &alpha;</div>
          </div>
        `;
        list.appendChild(item);
      });
    } catch (e) {
      list.innerHTML = `<div style="color: var(--accent-danger);">Error: ${e.message}</div>`;
    }
  }
});
