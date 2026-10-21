// Shared helpers: DOM builder, API client, formatting, the heat ramp and the tooltip.

export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export const $ = (sel, root = document) => root.querySelector(sel);

export const api = {
  async get(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`);
    return r.json();
  },
  async post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`);
    return r.json();
  },
  /** POST and consume a text/event-stream, calling onEvent(obj) per `data:` line. */
  async stream(url, body, onEvent, signal) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i);
        buf = buf.slice(i + 2);
        for (const line of chunk.split("\n")) if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
      }
    }
  },
};

export const fmt = {
  int: (n) => (n == null ? "—" : Math.round(n).toLocaleString("en-US")),
  compact(n) {
    if (n == null) return "—";
    const a = Math.abs(n);
    if (a >= 1e9) return (n / 1e9).toFixed(a >= 1e10 ? 0 : 1) + "B";
    if (a >= 1e6) return (n / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
    if (a >= 1e3) return (n / 1e3).toFixed(a >= 1e4 ? 0 : 1) + "K";
    return String(Math.round(n * 100) / 100);
  },
  fixed: (n, d = 2) => (n == null || Number.isNaN(n) ? "—" : Number(n).toFixed(d)),
  pct: (n, d = 1) => (n == null ? "—" : (n * 100).toFixed(d) + "%"),
  dur(s) {
    if (s == null) return "—";
    if (s < 1) return Math.round(s * 1000) + " ms";
    if (s < 90) return s.toFixed(1) + " s";
    const m = s / 60;
    return m < 90 ? m.toFixed(1) + " min" : (m / 60).toFixed(1) + " h";
  },
};

/** Make whitespace visible in a token string. */
export function showToken(text) {
  return text.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "→");
}

/** Append a token's text, rendering spaces as faint middots when `visible` is true. */
export function tokenNodes(text, visible = false) {
  if (!visible) return [document.createTextNode(text)];
  const out = [];
  for (const part of text.split(/( +|\n)/)) {
    if (!part) continue;
    if (/^ +$/.test(part)) out.push(h("span", { class: "sp" }, "·".repeat(part.length)));
    else if (part === "\n") out.push(h("span", { class: "sp" }, "↵\n"));
    else out.push(document.createTextNode(part));
  }
  return out;
}

// The heat ramp: cold iron -> cherry -> orange -> white-hot. Used for every intensity.
const RAMP = ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c", "#d9520f", "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"];
const RGB = RAMP.map((c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)));

export function heatRGB(t) {
  const x = Math.max(0, Math.min(1, t)) * (RGB.length - 1);
  const i = Math.min(RGB.length - 2, Math.floor(x));
  const f = x - i;
  return RGB[i].map((v, j) => Math.round(v + (RGB[i + 1][j] - v) * f));
}
export function heat(t, alpha = 1) {
  const [r, g, b] = heatRGB(t);
  return alpha === 1 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${alpha})`;
}
/** 256-entry RGBA lookup table for canvas heatmaps. */
export const HEAT_LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = heatRGB(i / 255);
    lut.set([r, g, b, 255], i * 4);
  }
  return lut;
})();

// ---------------------------------------------------------------- tooltip

const tip = () => document.getElementById("tooltip");
export function showTip(x, y, content) {
  const t = tip();
  t.replaceChildren(...(Array.isArray(content) ? content : [content]));
  t.classList.add("on");
  const pad = 14;
  const r = t.getBoundingClientRect();
  let left = x + pad, top = y + pad;
  if (left + r.width > innerWidth - 8) left = x - r.width - pad;
  if (top + r.height > innerHeight - 8) top = y - r.height - pad;
  t.style.left = left + "px";
  t.style.top = top + "px";
}
export function hideTip() { tip().classList.remove("on"); }
export function ttRow(label, value, color) {
  return h("div", { class: "tt-row" }, h("span", {}, color ? h("i", { style: { background: color } }) : null, label), h("b", {}, value));
}

/** Keep a range input's filled track in sync with its value. */
export function syncRange(input) {
  const set = () => {
    const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.setProperty("--pct", pct + "%");
  };
  input.addEventListener("input", set);
  set();
}

export function debounce(fn, ms = 200) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export function viewHead(title, desc, readouts) {
  return h("header", { class: "view-head" },
    h("div", {}, h("h1", { class: "view-title", html: title }), h("p", { class: "view-desc" }, desc)),
    readouts || h("div"));
}

export function readout(label, id, value = "—") {
  return h("div", { class: "readout" }, h("div", { class: "readout-label" }, label), h("div", { class: "readout-value", id }, value));
}
