// NodeSense Force-Directed Canvas Graph Visualizer
class GraphVisualizer {
  constructor(canvasId, onNodeSelected) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.onNodeSelected = onNodeSelected;

    this.nodes = [];
    this.edges = [];
    this.nodeMap = new Map();

    this.scale = 1.0;
    this.panX = 0;
    this.panY = 0;

    this.isDragging = false;
    this.dragNode = null;
    this.lastMousePos = { x: 0, y: 0 };
    this.hoverNode = null;

    this.animationFrame = null;
    this.initEvents();
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = rect.height;
    if (this.nodes.length === 0) {
      this.panX = this.canvas.width / 2;
      this.panY = this.canvas.height / 2;
    }
  }

  setData(subgraphData) {
    this.nodes = [];
    this.edges = [];
    this.nodeMap.clear();

    const w = this.canvas.width;
    const h = this.canvas.height;
    const centerId = subgraphData.center_id;

    // Place center node at center
    subgraphData.nodes.forEach((n, idx) => {
      let x, y;
      if (n.is_center) {
        x = 0;
        y = 0;
      } else {
        const angle = (idx / subgraphData.nodes.length) * 2 * Math.PI;
        const radius = 100 + Math.random() * 120;
        x = Math.cos(angle) * radius;
        y = Math.sin(angle) * radius;
      }

      const nodeObj = {
        ...n,
        x,
        y,
        vx: 0,
        vy: 0,
        radius: n.is_center ? 14 : 9
      };
      this.nodes.push(nodeObj);
      this.nodeMap.set(n.id, nodeObj);
    });

    subgraphData.edges.forEach(e => {
      const srcNode = this.nodeMap.get(e.source);
      const dstNode = this.nodeMap.get(e.target);
      if (srcNode && dstNode) {
        this.edges.push({ source: srcNode, target: dstNode });
      }
    });

    this.panX = w / 2;
    this.panY = h / 2;
    this.scale = 1.0;

    this.startSimulation();
  }

  startSimulation() {
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame);
    let stepCount = 0;

    const loop = () => {
      if (stepCount < 180 || this.isDragging) {
        this.simulatePhysics();
        stepCount++;
      }
      this.render();
      this.animationFrame = requestAnimationFrame(loop);
    };
    loop();
  }

  simulatePhysics() {
    const kRepel = 2400;
    const kSpring = 0.04;
    const springLen = 65;
    const damping = 0.85;

    // Repulsion between all node pairs
    for (let i = 0; i < this.nodes.length; i++) {
      for (let j = i + 1; j < this.nodes.length; j++) {
        const u = this.nodes[i];
        const v = this.nodes[j];
        let dx = v.x - u.x;
        let dy = v.y - u.y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
        if (dist < 300) {
          let force = kRepel / (dist * dist);
          let fx = (dx / dist) * force;
          let fy = (dy / dist) * force;
          if (u !== this.dragNode) { u.vx -= fx; u.vy -= fy; }
          if (v !== this.dragNode) { v.vx += fx; v.vy += fy; }
        }
      }
    }

    // Spring attraction along citation edges
    for (const e of this.edges) {
      const u = e.source;
      const v = e.target;
      let dx = v.x - u.x;
      let dy = v.y - u.y;
      let dist = Math.sqrt(dx * dx + dy * dy) || 1;
      let disp = dist - springLen;
      let fx = (dx / dist) * disp * kSpring;
      let fy = (dy / dist) * disp * kSpring;

      if (u !== this.dragNode) { u.vx += fx; u.vy += fy; }
      if (v !== this.dragNode) { v.vx -= fx; v.vy -= fy; }
    }

    // Center gravity & update positions
    for (const n of this.nodes) {
      if (n === this.dragNode) continue;
      // Gentle center pull
      n.vx -= n.x * 0.005;
      n.vy -= n.y * 0.005;

      n.vx *= damping;
      n.vy *= damping;

      n.x += n.vx;
      n.y += n.vy;
    }
  }

  render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.scale, this.scale);

    // Draw citation edges
    ctx.lineWidth = 1;
    for (const e of this.edges) {
      ctx.beginPath();
      ctx.moveTo(e.source.x, e.source.y);
      ctx.lineTo(e.target.x, e.target.y);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
      ctx.stroke();
    }

    // Draw nodes
    for (const n of this.nodes) {
      const isHovered = (n === this.hoverNode);
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.radius + (isHovered ? 3 : 0), 0, 2 * Math.PI);
      ctx.fillStyle = n.color || '#6366f1';
      ctx.fill();

      if (n.is_center) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Pulsing outer halo for center node
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius + 6, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      } else {
        ctx.strokeStyle = isHovered ? '#ffffff' : 'rgba(0,0,0,0.5)';
        ctx.lineWidth = isHovered ? 2 : 1;
        ctx.stroke();
      }

      // Draw label if center or hovered
      if (n.is_center || isHovered) {
        ctx.font = '11px sans-serif';
        ctx.fillStyle = '#f3f4f6';
        ctx.textAlign = 'center';
        ctx.fillText(`P#${n.id}`, n.x, n.y + n.radius + 14);
      }
    }

    ctx.restore();
  }

  screenToWorld(sx, sy) {
    return {
      x: (sx - this.panX) / this.scale,
      y: (sy - this.panY) / this.scale
    };
  }

  findNodeAt(wx, wy) {
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i];
      const dx = n.x - wx;
      const dy = n.y - wy;
      if (dx * dx + dy * dy <= (n.radius + 5) * (n.radius + 5)) {
        return n;
      }
    }
    return null;
  }

  initEvents() {
    this.canvas.addEventListener('mousedown', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const wPos = this.screenToWorld(sx, sy);

      const hit = this.findNodeAt(wPos.x, wPos.y);
      if (hit) {
        this.dragNode = hit;
        this.isDragging = true;
      } else {
        this.isDragging = true;
        this.dragNode = null;
      }
      this.lastMousePos = { x: sx, y: sy };
    });

    window.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const wPos = this.screenToWorld(sx, sy);

      this.hoverNode = this.findNodeAt(wPos.x, wPos.y);

      if (this.isDragging) {
        const dx = sx - this.lastMousePos.x;
        const dy = sy - this.lastMousePos.y;
        if (this.dragNode) {
          this.dragNode.x += dx / this.scale;
          this.dragNode.y += dy / this.scale;
        } else {
          this.panX += dx;
          this.panY += dy;
        }
        this.lastMousePos = { x: sx, y: sy };
      }
    });

    window.addEventListener('mouseup', (e) => {
      if (this.isDragging && this.dragNode && this.onNodeSelected) {
        const rect = this.canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        if (Math.abs(sx - this.lastMousePos.x) < 3 && Math.abs(sy - this.lastMousePos.y) < 3) {
          this.onNodeSelected(this.dragNode.id);
        }
      }
      this.isDragging = false;
      this.dragNode = null;
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;

      this.panX = sx - (sx - this.panX) * zoomFactor;
      this.panY = sy - (sy - this.panY) * zoomFactor;
      this.scale *= zoomFactor;
      this.scale = Math.max(0.2, Math.min(this.scale, 4.0));
    });
  }

  resetView() {
    this.scale = 1.0;
    this.panX = this.canvas.width / 2;
    this.panY = this.canvas.height / 2;
  }
}
