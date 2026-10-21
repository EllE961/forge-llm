// Attention explorer: a T×T heatmap for one head, a layer×head atlas of thumbnails, and
// per-head behavioural scores (previous-token, induction, attention-sink, entropy).

import { api, fmt, h, HEAT_LUT, hideTip, readout, showTip, showToken, ttRow, viewHead } from "../util.js";

const DEFAULT = "Lily had a red ball and Tom had a blue kite. Lily gave the red ball to Tom, and Tom gave the blue kite to Lily.";

export function mount(root, ctx) {
  const st = { layer: 0, head: 0, data: null, heads: null, hover: null };
  const input = h("input", { type: "text", value: DEFAULT });
  const runBtn = h("button", { class: "btn primary", onclick: () => load() }, "Trace attention");
  const chips = h("div", { class: "prompt-chips" });
  const canvas = h("canvas");
  const wrap = h("div", { class: "heat-canvas-wrap" }, canvas);
  const grid = h("div", { class: "head-grid" });
  const atlas = h("div", { class: "chart" });
  const title = h("div", { class: "panel-title" });

  root.append(
    viewHead("Attention <em>explorer</em>",
      "Where each query token looks. Rows are queries, columns are keys (causal, so the upper triangle is empty). Colour is √p on the heat ramp so faint-but-real attention stays visible.",
      h("div", { class: "readouts" }, readout("Prev-token", "a-prev"), readout("Induction", "a-ind"), readout("Sink", "a-sink"), readout("Entropy", "a-ent"))),
    h("div", { class: "view-body stack" },
      h("div", { class: "panel" }, h("div", { class: "panel-body row" }, h("div", { class: "grow" }, input), runBtn),
        h("div", { class: "panel-body", style: { paddingTop: 0 } }, chips)),
      h("div", { class: "attn-wrap" },
        h("div", { class: "panel" },
          h("div", { class: "panel-head" }, title,
            h("div", { class: "heat-scale" }, h("span", {}, "0"), h("div", { class: "bar" }), h("span", {}, "1"), h("span", {}, "√p"))),
          wrap),
        h("div", { class: "stack" },
          h("div", { class: "panel" },
            h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "All heads"), h("span", { class: "faint", style: { fontSize: "11px" } }, "rows = layers")),
            h("div", { class: "panel-body" }, grid)),
          h("div", { class: "panel" },
            h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "Head atlas"),
              h("span", { class: "faint", style: { fontSize: "11px" } }, "scored on synthetic probes")),
            h("div", { class: "panel-body" }, atlas))))));

  if (!ctx.status?.model) {
    wrap.replaceChildren(h("div", { class: "empty" }, h("div", {}, h("b", {}, "No checkpoint loaded"), "Train a model first.")));
    return;
  }
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") load(); });
  api.get("/api/heads").then((r) => { st.heads = r; pickStrongest(); drawAtlas(); updateReadouts(); if (st.data) { drawMain(); drawGrid(); } }).catch(() => {});
  load();

  function pickStrongest() {
    const best = st.heads.heads.reduce((a, b) => (b.induction > a.induction ? b : a));
    st.layer = best.layer; st.head = best.head;
  }

  async function load() {
    runBtn.disabled = true;
    try {
      const r = await api.post("/api/attention", { text: input.value });
      const bin = atob(r.data);
      const arr = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
      st.data = { ...r, arr };
      chips.replaceChildren(...r.tokens.map((t, i) => h("span", { "data-i": i }, showToken(t))));
      drawMain(); drawGrid();
    } catch (e) {
      wrap.replaceChildren(h("div", { class: "error-note" }, e.message));
    } finally { runBtn.disabled = false; }
  }

  const cellOf = (l, hd) => {
    const { T, n_heads, arr } = st.data;
    const off = (l * n_heads + hd) * T * T;
    return arr.subarray(off, off + T * T);
  };
  function toImage(vals, T) {
    const img = new ImageData(T, T);
    for (let i = 0; i < T * T; i++) {
      const row = Math.floor(i / T), col = i % T;
      if (col > row) { img.data.set([13, 12, 11, 255], i * 4); continue; } // masked (future)
      img.data.set(HEAT_LUT.subarray(vals[i] * 4, vals[i] * 4 + 4), i * 4);
    }
    const c = document.createElement("canvas");
    c.width = c.height = T;
    c.getContext("2d").putImageData(img, 0, 0);
    return c;
  }

  function drawMain() {
    const { T, tokens } = st.data;
    title.replaceChildren(h("b", {}, `Layer ${st.layer} · head ${st.head}`), h("span", { class: "faint" }, `  ·  ${T} tokens`));
    const left = 96, bottom = 80;
    // fit the matrix to the panel width *and* keep it on screen vertically
    const avail = Math.max(240, Math.min(wrap.clientWidth - 40 - left, 560, innerHeight - 440));
    const cell = Math.max(4, Math.floor(avail / T));
    const size = cell * T;
    const dpr = devicePixelRatio || 1;
    canvas.width = (left + size + 8) * dpr; canvas.height = (size + bottom) * dpr;
    canvas.style.width = left + size + 8 + "px"; canvas.style.height = size + bottom + "px";
    const g = canvas.getContext("2d");
    g.scale(dpr, dpr);
    g.imageSmoothingEnabled = false;
    g.drawImage(toImage(cellOf(st.layer, st.head), T), left, 0, size, size);
    // cell gaps on large cells read as a grid of tiles rather than a blur
    if (cell >= 10) {
      g.strokeStyle = "rgba(13,12,11,0.9)"; g.lineWidth = 1;
      for (let i = 1; i < T; i++) {
        g.beginPath(); g.moveTo(left + i * cell + 0.5, 0); g.lineTo(left + i * cell + 0.5, size); g.stroke();
        g.beginPath(); g.moveTo(left, i * cell + 0.5); g.lineTo(left + size, i * cell + 0.5); g.stroke();
      }
    }
    const fs = Math.min(11, Math.max(8, cell - 1));
    g.font = `500 ${fs}px "JetBrains Mono", monospace`;
    g.textBaseline = "middle";
    tokens.forEach((t, i) => {
      const lab = showToken(t).slice(0, 12);
      const hot = st.hover && (st.hover.i === i);
      g.fillStyle = hot ? "#ede7de" : "#7a736c";
      g.textAlign = "right";
      g.fillText(lab, left - 6, i * cell + cell / 2);
      g.save();
      g.translate(left + i * cell + cell / 2, size + 6);
      g.rotate(-Math.PI / 3);
      g.textAlign = "right";
      g.fillStyle = st.hover && st.hover.j === i ? "#ede7de" : "#7a736c";
      g.fillText(lab, 0, 0);
      g.restore();
    });
    if (st.hover) {
      const { i, j } = st.hover;
      g.strokeStyle = "rgba(237,231,222,0.55)"; g.lineWidth = 1;
      g.strokeRect(left + 0.5, i * cell + 0.5, size - 1, cell - 1);
      g.strokeRect(left + j * cell + 0.5, 0.5, cell - 1, size - 1);
      g.strokeStyle = "#ede7de"; g.lineWidth = 2;
      g.strokeRect(left + j * cell + 1, i * cell + 1, cell - 2, cell - 2);
    }
    canvas.onmousemove = (ev) => {
      const r = canvas.getBoundingClientRect();
      const x = ev.clientX - r.left - left, y = ev.clientY - r.top;
      const i = Math.floor(y / cell), j = Math.floor(x / cell);
      if (x < 0 || y < 0 || i >= T || j >= T || j > i) { leave(); return; }
      if (!st.hover || st.hover.i !== i || st.hover.j !== j) { st.hover = { i, j }; drawMain(); markChips(i, j); }
      const v = cellOf(st.layer, st.head)[i * T + j] / 255;
      showTip(ev.clientX, ev.clientY, [h("div", { class: "tt-title" }, `"${showToken(tokens[i])}" → "${showToken(tokens[j])}"`),
        ttRow("query pos", i), ttRow("key pos", j), ttRow("weight", fmt.pct(v * v, 1))]);
    };
    canvas.onmouseleave = leave;
  }
  function leave() { hideTip(); if (st.hover) { st.hover = null; drawMain(); markChips(-1, -1); } }
  function markChips(i, j) {
    chips.querySelectorAll("span").forEach((s, n) => { s.classList.toggle("q", n === i); s.classList.toggle("k", n === j && n !== i); });
  }

  function badge(l, hd) {
    const s = st.heads?.heads.find((x) => x.layer === l && x.head === hd);
    if (!s) return null;
    if (s.induction > 0.3) return "IND";
    if (s.prev > 0.5) return "PREV";
    if (s.sink > 0.6) return "SINK";
    return null;
  }

  function drawGrid() {
    if (!st.data) return;
    const { n_layers, n_heads, T } = st.data;
    grid.style.setProperty("--heads", n_heads);
    const cells = [h("span"), ...Array.from({ length: n_heads }, (_, i) => h("span", { class: "lbl" }, `h${i}`))];
    for (let l = 0; l < n_layers; l++) {
      cells.push(h("span", { class: "lbl" }, `L${l}`));
      for (let hd = 0; hd < n_heads; hd++) {
        const c = toImage(cellOf(l, hd), T);
        const b = badge(l, hd);
        const cell = h("div", { class: `head-cell ${l === st.layer && hd === st.head ? "on" : ""}`, title: `layer ${l} head ${hd}`,
          onclick: () => { st.layer = l; st.head = hd; drawMain(); drawGrid(); updateReadouts(); drawAtlas(); } }, c, b ? h("span", { class: "badge" }, b) : null);
        cells.push(cell);
      }
    }
    grid.replaceChildren(...cells);
  }

  function updateReadouts() {
    const s = st.heads?.heads.find((x) => x.layer === st.layer && x.head === st.head);
    if (!s) return;
    document.getElementById("a-prev").textContent = fmt.pct(s.prev, 0);
    document.getElementById("a-ind").textContent = fmt.pct(s.induction, 0);
    document.getElementById("a-sink").textContent = fmt.pct(s.sink, 0);
    document.getElementById("a-ent").innerHTML = `${(s.entropy / Math.LN2).toFixed(2)}<small>bits</small>`;
  }

  function drawAtlas() {
    if (!st.heads) return;
    const NS = "http://www.w3.org/2000/svg";
    const W = atlas.clientWidth || 320, H = 220, m = { l: 40, r: 10, t: 10, b: 30 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("height", H);
    const add = (tag, attrs, text) => {
      const n = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
      if (text !== undefined) n.textContent = text;
      svg.append(n); return n;
    };
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      const y = m.t + ih - v * ih, x = m.l + v * iw;
      add("line", { class: "gridline", x1: m.l, x2: m.l + iw, y1: Math.round(y) + 0.5, y2: Math.round(y) + 0.5 });
      add("text", { x: m.l - 8, y: y + 3.5, "text-anchor": "end", class: "tick", fill: "var(--ash-3)", "font-size": 10.5, "font-family": "var(--font-mono)" }, v.toFixed(2));
      add("text", { x, y: H - 12, "text-anchor": "middle", fill: "var(--ash-3)", "font-size": 10.5, "font-family": "var(--font-mono)" }, v.toFixed(2));
    }
    add("text", { x: m.l + iw, y: H - 1, "text-anchor": "end", fill: "var(--ash-3)", "font-size": 10.5, "font-family": "var(--font-mono)" }, "previous-token score →");
    add("text", { x: m.l + 4, y: m.t + 8, fill: "var(--ash-3)", "font-size": 10.5, "font-family": "var(--font-mono)" }, "↑ induction score");
    for (const s of st.heads.heads) {
      const on = s.layer === st.layer && s.head === st.head;
      const c = add("circle", { cx: m.l + Math.min(1, s.prev) * iw, cy: m.t + ih - Math.min(1, s.induction) * ih, r: on ? 6 : 4,
        fill: on ? "var(--ember)" : "var(--series-2)", stroke: "var(--iron)", "stroke-width": 2, opacity: on ? 1 : 0.75, style: "cursor:pointer" });
      c.addEventListener("mousemove", (ev) => showTip(ev.clientX, ev.clientY, [h("div", { class: "tt-title" }, `L${s.layer} · h${s.head}`),
        ttRow("prev-token", fmt.pct(s.prev, 0)), ttRow("induction", fmt.pct(s.induction, 0)), ttRow("sink", fmt.pct(s.sink, 0))]));
      c.addEventListener("mouseleave", hideTip);
      c.addEventListener("click", () => { st.layer = s.layer; st.head = s.head; drawMain(); drawGrid(); updateReadouts(); drawAtlas(); });
    }
    atlas.replaceChildren(svg);
  }
}
