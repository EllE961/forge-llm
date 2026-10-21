// Embedding map: the 8192 learned token vectors in 2-D (t-SNE over the top-50 principal
// components, or plain PCA). Pan with drag, zoom with the wheel, search, click for neighbours.

import { api, fmt, h, heat, hideTip, readout, showTip, showToken, ttRow, viewHead } from "../util.js";

export function mount(root, ctx) {
  const st = { data: null, view: { x: 0, y: 0, k: 1 }, hover: -1, sel: -1, nbrs: [], matches: new Set(), colorBy: "norm",
    method: "tsne" };
  const canvas = h("canvas");
  const search = h("input", { type: "text", placeholder: "Highlight tokens containing…  (try: ing)" });
  const colorSeg = h("div", { class: "seg text" },
    h("button", { class: "on", onclick: (e) => setColor("norm", e.target) }, "‖e‖ norm"),
    h("button", { onclick: (e) => setColor("kind", e.target) }, "word start"));
  const methodSeg = h("div", { class: "seg text" },
    h("button", { class: "on", onclick: (e) => setMethod("tsne", e.target) }, "t-SNE"),
    h("button", { onclick: (e) => setMethod("pca", e.target) }, "PCA"));
  const status = h("div", { class: "emb-status" });
  const wrapEl = h("div", { class: "emb-wrap" }, canvas, h("div", { class: "emb-hud" }, search, methodSeg, colorSeg), status);
  const side = h("div", { class: "panel-body stack" });
  const legend = h("div", { class: "panel-body", style: { borderTop: "1px solid var(--seam-soft)" } });

  root.append(
    viewHead("Embedding <em>map</em>",
      "Each dot is one of the 8,192 learned token embeddings. Tokens the model uses alike end up together — names, feelings, actions and punctuation form their own neighbourhoods.",
      h("div", { class: "readouts" }, readout("Tokens", "e-n"), readout("PCA variance (2 PCs)", "e-var"), readout("Selected", "e-sel"))),
    h("div", { class: "view-body" },
      h("div", { class: "split" },
        h("div", { class: "panel", style: { overflow: "hidden" } }, wrapEl),
        h("div", { class: "panel" },
          h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "Nearest neighbours"), h("span", { class: "faint", style: { fontSize: "11px" } }, "cosine, full 512-d space")),
          side, legend))));

  side.replaceChildren(h("div", { class: "empty", style: { minHeight: "160px" } }, h("div", {}, h("b", {}, "Click a point"), "to list its closest tokens.")));
  if (!ctx.status?.model) return;

  load();

  async function load() {
    status.replaceChildren(h("span", { class: "chip live" }, h("i"),
      st.method === "tsne" ? "computing t-SNE · ~1 min on first load, cached after" : "projecting…"));
    try {
      const d = await api.get(`/api/embeddings?method=${st.method}`);
      st.data = d;
      const nmax = Math.max(...d.norm), nmin = Math.min(...d.norm);
      st.normT = d.norm.map((v) => (v - nmin) / (nmax - nmin || 1));
      document.getElementById("e-n").textContent = fmt.int(d.x.length);
      document.getElementById("e-var").textContent = fmt.pct(d.explained_variance[0] + d.explained_variance[1], 1);
      status.replaceChildren();
      fit();
      drawLegend();
      draw();
      if (st.sel < 0) {
        const i = d.tokens.indexOf(" happy");
        if (i >= 0) select(i);
      }
    } catch (e) {
      status.replaceChildren(h("span", { class: "error-note" }, e.message));
    }
  }
  function setMethod(m, btn) {
    if (m === st.method) return;
    st.method = m;
    methodSeg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
    load();
  }

  function setColor(mode, btn) {
    st.colorBy = mode;
    colorSeg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
    drawLegend(); draw();
  }
  function drawLegend() {
    legend.replaceChildren(st.colorBy === "norm"
      ? h("div", { class: "heat-scale" }, h("span", {}, "small"), h("div", { class: "bar" }), h("span", {}, "large ‖e‖"))
      : h("div", { class: "legend" },
          h("span", {}, h("i", { class: "key", style: { background: "var(--series-1)", borderRadius: "50%" } }), "starts a word (leading space)"),
          h("span", {}, h("i", { class: "key", style: { background: "var(--series-2)", borderRadius: "50%" } }), "word piece / punctuation")));
  }

  function fit() {
    const { x, y } = st.data;
    const xs = [...x].sort((a, b) => a - b), ys = [...y].sort((a, b) => a - b);
    const q = (arr, p) => arr[Math.floor(p * (arr.length - 1))];
    st.bounds = { x0: q(xs, 0.005), x1: q(xs, 0.995), y0: q(ys, 0.005), y1: q(ys, 0.995) };
    st.view = { x: 0, y: 0, k: 1 };
  }
  function project(i, W, H) {
    const b = st.bounds, pad = 40;
    const s = Math.min((W - 2 * pad) / (b.x1 - b.x0), (H - 2 * pad) / (b.y1 - b.y0));
    const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
    const px = W / 2 + (st.data.x[i] - cx) * s, py = H / 2 - (st.data.y[i] - cy) * s;
    return [(px - W / 2) * st.view.k + W / 2 + st.view.x, (py - H / 2) * st.view.k + H / 2 + st.view.y];
  }

  function draw() {
    if (!st.data) return;
    const W = wrapEl.clientWidth, H = wrapEl.clientHeight, dpr = devicePixelRatio || 1;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const g = canvas.getContext("2d");
    g.scale(dpr, dpr);
    g.fillStyle = "#0b0a09"; g.fillRect(0, 0, W, H);
    const n = st.data.x.length, toks = st.data.tokens;
    const hasMatch = st.matches.size > 0;
    const r = Math.max(1.3, Math.min(3.2, 1.6 * Math.sqrt(st.view.k)));
    for (let i = 0; i < n; i++) {
      const [px, py] = project(i, W, H);
      if (px < -5 || py < -5 || px > W + 5 || py > H + 5) continue;
      let col;
      if (st.colorBy === "norm") col = heat(0.18 + 0.82 * st.normT[i], hasMatch ? 0.25 : 0.85);
      else col = toks[i].startsWith(" ") ? (hasMatch ? "rgba(217,89,38,0.22)" : "rgba(217,89,38,0.8)") : (hasMatch ? "rgba(57,135,229,0.22)" : "rgba(57,135,229,0.8)");
      g.fillStyle = col;
      g.beginPath(); g.arc(px, py, r, 0, Math.PI * 2); g.fill();
    }
    // search matches on top, labelled when zoomed or few
    if (hasMatch) {
      g.font = '500 11px "JetBrains Mono", monospace';
      const few = st.matches.size <= 60;
      for (const i of st.matches) {
        const [px, py] = project(i, W, H);
        g.fillStyle = "#fccf6b"; g.strokeStyle = "#0b0a09"; g.lineWidth = 2;
        g.beginPath(); g.arc(px, py, r + 1.6, 0, Math.PI * 2); g.stroke(); g.fill();
        if (few || st.view.k > 3) { g.fillStyle = "#ede7de"; g.fillText(showToken(toks[i]), px + 6, py - 6); }
      }
    }
    if (st.sel >= 0) {
      const [sx, sy] = project(st.sel, W, H);
      g.lineWidth = 1;
      for (const [rank, nb] of st.nbrs.entries()) {
        const [nx, ny] = project(nb.id, W, H);
        g.strokeStyle = `rgba(246,199,94,${0.15 + 0.6 * Math.max(0, nb.cos)})`;
        g.beginPath(); g.moveTo(sx, sy); g.lineTo(nx, ny); g.stroke();
        g.fillStyle = "#f6c75e"; g.beginPath(); g.arc(nx, ny, r + 1.5, 0, Math.PI * 2); g.fill();
        if (rank < 6) { g.fillStyle = "#aaa299"; g.font = '500 11px "JetBrains Mono", monospace'; g.fillText(showToken(nb.text), nx + 6, ny + 4); }
      }
      g.fillStyle = "#f2711c"; g.strokeStyle = "#0b0a09"; g.lineWidth = 2;
      g.beginPath(); g.arc(sx, sy, 6, 0, Math.PI * 2); g.stroke(); g.fill();
      g.fillStyle = "#ede7de"; g.font = '600 13px "JetBrains Mono", monospace'; g.fillText(showToken(toks[st.sel]), sx + 9, sy - 8);
    }
    if (st.hover >= 0 && st.hover !== st.sel) {
      const [hx, hy] = project(st.hover, W, H);
      g.strokeStyle = "#ede7de"; g.lineWidth = 1.5;
      g.beginPath(); g.arc(hx, hy, r + 3, 0, Math.PI * 2); g.stroke();
    }
  }

  function nearest(mx, my) {
    const W = wrapEl.clientWidth, H = wrapEl.clientHeight;
    let best = -1, bd = 14 * 14;
    for (let i = 0; i < st.data.x.length; i++) {
      const [px, py] = project(i, W, H);
      const d = (px - mx) ** 2 + (py - my) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  let drag = null;
  canvas.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY, vx: st.view.x, vy: st.view.y, moved: false }; });
  addEventListener("mouseup", (e) => {
    if (drag && !drag.moved && st.data) {
      const r = canvas.getBoundingClientRect();
      const i = nearest(e.clientX - r.left, e.clientY - r.top);
      if (i >= 0) select(i);
    }
    drag = null;
  });
  canvas.addEventListener("mousemove", (e) => {
    if (!st.data) return;
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      st.view.x = drag.vx + dx; st.view.y = drag.vy + dy;
      draw(); return;
    }
    const r = canvas.getBoundingClientRect();
    const i = nearest(e.clientX - r.left, e.clientY - r.top);
    if (i !== st.hover) { st.hover = i; draw(); }
    if (i >= 0) showTip(e.clientX, e.clientY, [h("div", { class: "tt-title" }, `"${showToken(st.data.tokens[i])}"`), ttRow("id", i), ttRow("‖e‖", st.data.norm[i].toFixed(2))]);
    else hideTip();
  });
  canvas.addEventListener("mouseleave", () => { hideTip(); st.hover = -1; draw(); });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top, W = wrapEl.clientWidth, H = wrapEl.clientHeight;
    const f = Math.exp(-e.deltaY * 0.0015);
    const k = Math.max(0.5, Math.min(40, st.view.k * f));
    const ratio = k / st.view.k;
    st.view.x = mx - W / 2 - (mx - W / 2 - st.view.x) * ratio;
    st.view.y = my - H / 2 - (my - H / 2 - st.view.y) * ratio;
    st.view.k = k;
    draw();
  }, { passive: false });
  new ResizeObserver(() => draw()).observe(wrapEl);
  search.addEventListener("input", () => {
    const q = search.value;
    st.matches = new Set();
    if (q && st.data) st.data.tokens.forEach((t, i) => { if (t.toLowerCase().includes(q.toLowerCase())) st.matches.add(i); });
    draw();
  });

  async function select(i) {
    st.sel = i;
    // only pan when the selection would be off-screen (e.g. picked from the neighbour list)
    const W = wrapEl.clientWidth, H = wrapEl.clientHeight;
    const [px, py] = project(i, W, H);
    if (px < 20 || px > W - 80 || py < 20 || py > H - 20) { st.view.x += W / 2 - px; st.view.y += H / 2 - py; }
    document.getElementById("e-sel").textContent = `"${showToken(st.data.tokens[i])}"`;
    draw();
    try {
      const r = await api.get(`/api/neighbors/${i}?k=12`);
      st.nbrs = r.neighbors;
      side.replaceChildren(
        h("div", { class: "inspector-token" }, `"${showToken(r.text)}"`),
        h("div", { class: "neighbors" }, ...r.neighbors.map((nb) =>
          h("div", { class: "nb", onclick: () => select(nb.id) }, h("span", {}, `"${showToken(nb.text)}"`), h("span", {}, nb.cos.toFixed(3))))));
      draw();
    } catch (e) { side.replaceChildren(h("div", { class: "error-note" }, e.message)); }
  }
}

export function onShow() { window.dispatchEvent(new Event("resize")); }
