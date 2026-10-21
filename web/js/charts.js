// Hand-rolled SVG charts: thin 2px lines, hairline solid grid, crosshair + tooltip on hover,
// one y-axis only. Re-render on resize.

import { h, hideTip, showTip, ttRow } from "./util.js";

const NS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) n.setAttribute(k, v);
  return n;
};

export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const step0 = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(step0));
  const err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
  return ticks;
}

function logTicks(min, max) {
  const out = [];
  const mantissas = max / min < 20 ? [1, 1.5, 2, 3, 4, 5, 7] : [1, 2, 5];
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) {
    for (const m of mantissas) {
      const v = m * 10 ** e;
      if (v >= min * 0.999 && v <= max * 1.001) out.push(v);
    }
  }
  return out.length >= 2 ? out : [min, max];
}

/**
 * lineChart(el, opts)
 *   series: [{ name, color, points: [{x, y, ...}], kind: "line" | "dots", area: bool, dim: bool }]
 *   x: { label, format, log }, y: { label, format, min, max, log }
 *   height, onHover(point|null, seriesIndex), tooltipTitle(x)
 */
export function lineChart(el, opts) {
  el.classList.add("chart");
  const draw = () => render(el, opts);
  draw();
  if (!el._ro) {
    el._ro = new ResizeObserver(() => { if (el._lastW !== el.clientWidth) draw(); });
    el._ro.observe(el);
  }
  el._update = (next) => { Object.assign(opts, next); draw(); };
  return el;
}

function render(el, o) {
  const W = el.clientWidth || 600;
  el._lastW = W;
  const H = o.height || 220;
  const m = { top: 12, right: o.marginRight ?? 16, bottom: 28, left: o.marginLeft ?? 48 };
  const iw = Math.max(10, W - m.left - m.right), ih = H - m.top - m.bottom;
  const all = o.series.flatMap((sr) => sr.points);
  if (!all.length) {
    el.replaceChildren(h("div", { class: "empty", style: { minHeight: H + "px" } }, o.emptyText || "No data yet"));
    return;
  }
  let xmin = Math.min(...all.map((p) => p.x)), xmax = Math.max(...all.map((p) => p.x));
  let ymin = o.y?.min ?? Math.min(...all.map((p) => p.y)), ymax = o.y?.max ?? Math.max(...all.map((p) => p.y));
  if (o.y?.min === undefined && !o.y?.log) { const pad = (ymax - ymin) * 0.06 || 1; ymin -= pad; ymax += pad; }
  if (o.x?.log) xmin = Math.max(xmin, 1e-9);
  const sx = o.x?.log
    ? (v) => m.left + ((Math.log(v) - Math.log(xmin)) / (Math.log(xmax) - Math.log(xmin) || 1)) * iw
    : (v) => m.left + ((v - xmin) / (xmax - xmin || 1)) * iw;
  const sy = o.y?.log
    ? (v) => m.top + ih - ((Math.log(v) - Math.log(ymin)) / (Math.log(ymax) - Math.log(ymin) || 1)) * ih
    : (v) => m.top + ih - ((v - ymin) / (ymax - ymin || 1)) * ih;
  const fx = o.x?.format || ((v) => v);
  const fy = o.y?.format || ((v) => v);

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": o.ariaLabel || "chart" });
  const yt = o.y?.log ? logTicks(ymin, ymax) : niceTicks(ymin, ymax, Math.max(3, Math.round(ih / 48)));
  for (const t of yt) {
    if (t < ymin - 1e-12 || t > ymax + 1e-12) continue;
    const y = Math.round(sy(t)) + 0.5;
    svg.append(s("line", { class: "gridline", x1: m.left, x2: m.left + iw, y1: y, y2: y }));
    const tx = s("text", { x: m.left - 8, y: y + 3.5, "text-anchor": "end", class: "tick" });
    tx.textContent = fy(t);
    const g = s("g", { class: "tick" }); g.append(tx); svg.append(g);
  }
  const xt = o.x?.log ? logTicks(xmin, xmax) : niceTicks(xmin, xmax, Math.max(3, Math.round(iw / 110)));
  const labelRoom = o.x?.label ? o.x.label.length * 6.6 + 18 : 0; // keep tick labels clear of the axis title
  for (const t of xt) {
    if (t < xmin - 1e-12 || t > xmax + 1e-12) continue;
    if (sx(t) > m.left + iw - labelRoom) continue;
    const g = s("g", { class: "tick" });
    const tx = s("text", { x: sx(t), y: m.top + ih + 18, "text-anchor": "middle" });
    tx.textContent = fx(t);
    g.append(tx); svg.append(g);
  }
  svg.append(s("line", { class: "baseline", x1: m.left, x2: m.left + iw, y1: m.top + ih + 0.5, y2: m.top + ih + 0.5 }));
  if (o.x?.label) {
    const t = s("text", { x: m.left + iw, y: H - 1, "text-anchor": "end", class: "axis" });
    t.textContent = o.x.label;
    const g = s("g", { class: "axis" }); g.append(t); svg.append(g);
  }

  const clipId = "c" + Math.random().toString(36).slice(2);
  const clip = s("clipPath", { id: clipId });
  clip.append(s("rect", { x: m.left, y: m.top - 4, width: iw, height: ih + 8 }));
  svg.append(clip);
  const plot = s("g", { "clip-path": `url(#${clipId})` });
  svg.append(plot);

  o.series.forEach((sr) => {
    const pts = sr.points.filter((p) => Number.isFinite(p.y) && (!o.y?.log || p.y > 0));
    if (!pts.length) return;
    if (sr.kind === "dots") {
      for (const p of pts) {
        plot.append(s("circle", { cx: sx(p.x), cy: sy(p.y), r: sr.r || 4, fill: sr.color, stroke: "var(--iron)", "stroke-width": 2, opacity: sr.dim ? 0.35 : 1 }));
      }
      return;
    }
    const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
    if (sr.area) {
      const base = sy(!o.y?.log && ymin <= 0 && ymax >= 0 ? 0 : ymin);
      plot.append(s("path", { d: `${d}L${sx(pts.at(-1).x)},${base}L${sx(pts[0].x)},${base}Z`, fill: sr.color, opacity: 0.1 }));
    }
    plot.append(s("path", {
      d, fill: "none", stroke: sr.color, "stroke-width": sr.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round",
      opacity: sr.dim ? 0.3 : 1,
    }));
    if (sr.endDot !== false && !sr.dim) {
      const p = pts.at(-1);
      plot.append(s("circle", { cx: sx(p.x), cy: sy(p.y), r: 4, fill: sr.color, stroke: "var(--iron)", "stroke-width": 2 }));
    }
  });

  // hover layer
  const cross = s("line", { class: "crosshair", y1: m.top, y2: m.top + ih, x1: 0, x2: 0 });
  const dots = o.series.map((sr) => s("circle", { r: 4.5, fill: sr.color, stroke: "var(--iron)", "stroke-width": 2, opacity: 0 }));
  svg.append(cross, ...dots);
  const hit = s("rect", { x: m.left, y: m.top, width: iw, height: ih, fill: "transparent" });
  svg.append(hit);
  const xs = [...new Set(all.map((p) => p.x))].sort((a, b) => a - b);
  hit.addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    let best = xs[0], bd = Infinity;
    for (const x of xs) { const d = Math.abs(sx(x) - px); if (d < bd) { bd = d; best = x; } }
    const X = Math.round(sx(best)) + 0.5;
    cross.setAttribute("x1", X); cross.setAttribute("x2", X);
    const rows = [h("div", { class: "tt-title" }, o.tooltipTitle ? o.tooltipTitle(best) : `${o.x?.label || "x"} ${fx(best)}`)];
    o.series.forEach((sr, i) => {
      let p = null, pd = Infinity;
      for (const q of sr.points) { const d = Math.abs(q.x - best); if (d < pd) { pd = d; p = q; } }
      const near = p && Math.abs(sx(p.x) - X) < Math.max(6, iw / Math.max(1, sr.points.length));
      dots[i].setAttribute("opacity", near && !sr.dim ? 1 : 0);
      if (p && near) {
        dots[i].setAttribute("cx", sx(p.x)); dots[i].setAttribute("cy", sy(p.y));
        rows.push(ttRow(sr.name, (sr.format || fy)(p.y), sr.color));
      }
    });
    showTip(ev.clientX, ev.clientY, rows);
    o.onHover?.(best);
  });
  hit.addEventListener("mouseleave", () => { hideTip(); dots.forEach((d) => d.setAttribute("opacity", 0)); o.onHover?.(null); });
  el.replaceChildren(svg);
}

/** Vertical columns from one baseline; 4px rounded data-end; value on hover. */
export function columnChart(el, { labels, values, color = "var(--series-1)", height = 160, format = (v) => v, title = (l) => l, highlight }) {
  el.classList.add("chart");
  const W = el.clientWidth || 300, H = height;
  const m = { top: 14, right: 8, bottom: 24, left: 36 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const max = Math.max(1e-9, ...values);
  const ticks = niceTicks(0, max, 3);
  const top = Math.max(max, ticks.at(-1));
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, height: H });
  for (const t of ticks) {
    const y = Math.round(m.top + ih - (t / top) * ih) + 0.5;
    svg.append(s("line", { class: "gridline", x1: m.left, x2: m.left + iw, y1: y, y2: y }));
    const tx = s("text", { x: m.left - 8, y: y + 3.5, "text-anchor": "end" });
    tx.textContent = format(t);
    const g = s("g", { class: "tick" }); g.append(tx); svg.append(g);
  }
  const band = iw / labels.length;
  const bw = Math.min(24, band - 2);
  labels.forEach((lab, i) => {
    const v = values[i];
    const x = m.left + i * band + (band - bw) / 2;
    const bh = (v / top) * ih;
    const y = m.top + ih - bh;
    const r = Math.min(4, bh / 2, bw / 2);
    const path = bh <= 0 ? "" : `M${x},${m.top + ih}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${m.top + ih}Z`;
    const col = highlight === undefined || highlight === i ? color : "var(--ash-4)";
    const bar = s("path", { d: path, fill: col });
    svg.append(bar);
    const t = s("text", { x: x + bw / 2, y: H - 6, "text-anchor": "middle" });
    t.textContent = lab;
    const g = s("g", { class: "tick" }); g.append(t); svg.append(g);
    const hit = s("rect", { x: m.left + i * band, y: m.top, width: band, height: ih, fill: "transparent" });
    hit.addEventListener("mousemove", (ev) => showTip(ev.clientX, ev.clientY, [h("div", { class: "tt-title" }, title(lab)), ttRow("value", format(v), col)]));
    hit.addEventListener("mouseleave", hideTip);
    svg.append(hit);
  });
  svg.append(s("line", { class: "baseline", x1: m.left, x2: m.left + iw, y1: m.top + ih + 0.5, y2: m.top + ih + 0.5 }));
  el.replaceChildren(svg);
}

/** Bias-corrected exponential moving average (as in Adam / TensorBoard smoothing). */
export function ema(points, alpha = 0.9) {
  let acc = 0;
  return points.map((p, i) => {
    acc = alpha * acc + (1 - alpha) * p.y;
    return { ...p, y: acc / (1 - alpha ** (i + 1)) };
  });
}
