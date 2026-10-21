// Generate: stream tokens over SSE; every token wears its heat (surprisal or entropy) and
// can be inspected for the model's full top-5 distribution at that step.

import { lineChart } from "../charts.js";
import { api, fmt, h, heat, readout, showToken, syncRange, viewHead } from "../util.js";

const PRESETS = ["Once upon a time", "Lily wanted a puppy, but her mom said", "The little robot looked at the stars and",
  "Tom was sad because", "One day, a dragon flew into the village. The dragon"];

const LN2 = Math.log(2);
const bits = (p) => -Math.log(Math.max(p, 1e-12)) / LN2;

export function mount(root, ctx) {
  const state = { tokens: [], mode: "surprisal", pinned: null, abort: null, running: false };

  const ro = h("div", { class: "readouts" },
    readout("Throughput", "g-tps"), readout("Tokens", "g-n"), readout("Mean surprisal", "g-bits"), readout("Perplexity", "g-ppl"));
  const prompt = h("textarea", { rows: 3, spellcheck: "false", placeholder: "Write the start of a story…" }, PRESETS[0]);
  const runBtn = h("button", { class: "btn primary", onclick: () => run() }, "Generate", h("kbd", {}, "Ctrl ↵"));
  const stopBtn = h("button", { class: "btn", disabled: true, onclick: () => state.abort?.abort() }, "Stop");
  const modeSeg = h("div", { class: "seg text", role: "tablist" },
    ...["surprisal", "entropy"].map((m) => h("button", { class: m === state.mode ? "on" : "", onclick: (e) => setMode(m, e.target) }, m === "surprisal" ? "Surprisal" : "Entropy")));
  const presets = h("div", { class: "row", style: { flexWrap: "wrap", gap: "6px" } },
    ...PRESETS.slice(1).map((p) => h("button", { class: "btn ghost", style: { height: "26px", fontWeight: 500, fontSize: "12px" }, onclick: () => { prompt.value = p; run(); } }, p.length > 34 ? p.slice(0, 32) + "…" : p)));
  const anvil = h("div", { class: "anvil", "aria-live": "polite" });
  const scaleLabel = h("span", {}, "0 → 10 bits");
  const heatScale = h("div", { class: "heat-scale" }, h("span", {}, "cool"), h("div", { class: "bar" }), h("span", {}, "hot"), scaleLabel);
  const trace = h("div");

  // sampling controls
  const ctl = (label, input, out) => h("div", {}, h("label", { class: "field-label" }, label, out), input);
  const range = (min, max, step, value, cls) => h("input", { type: "range", min, max, step, value, class: cls || "" });
  const temp = range(0, 1.6, 0.05, 0.8, "heat"), topP = range(0.05, 1, 0.01, 0.95), topK = range(0, 200, 1, 0);
  const maxTok = range(16, 512, 8, 200), rep = range(1, 1.6, 0.01, 1);
  const outs = [temp, topP, topK, maxTok, rep].map(() => h("output"));
  const seed = h("input", { type: "text", placeholder: "random", inputmode: "numeric" });
  const fmtCtl = () => {
    outs[0].textContent = (+temp.value).toFixed(2) + (+temp.value === 0 ? " · greedy" : "");
    outs[1].textContent = +topP.value >= 1 ? "off" : (+topP.value).toFixed(2);
    outs[2].textContent = +topK.value === 0 ? "off" : topK.value;
    outs[3].textContent = maxTok.value;
    outs[4].textContent = (+rep.value).toFixed(2);
  };
  [temp, topP, topK, maxTok, rep].forEach((r) => { syncRange(r); r.addEventListener("input", fmtCtl); });
  fmtCtl();

  const inspector = h("div", { class: "panel-body stack" });

  root.append(
    viewHead("Generate <em>token by token</em>",
      "Sampling from the 27M-parameter model trained here. Each token is underlined by how surprised the model was to emit it; hover one to see the distribution it was drawn from.", ro),
    h("div", { class: "view-body" },
      h("div", { class: "split" },
        h("div", { class: "stack" },
          h("div", { class: "panel" },
            h("div", { class: "panel-body stack" }, prompt,
              h("div", { class: "row" }, runBtn, stopBtn, h("div", { class: "grow" }), modeSeg)),
            h("div", { class: "panel-body", style: { paddingTop: 0 } }, presets)),
          h("div", { class: "panel" },
            h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "Output"), heatScale),
            anvil),
          h("div", { class: "panel" },
            h("div", { class: "panel-head" },
              h("div", { class: "panel-title" }, "Uncertainty trace"),
              h("div", { class: "legend" },
                h("span", {}, h("i", { class: "key line", style: { background: "var(--series-2)" } }), "entropy of next-token distribution"),
                h("span", {}, h("i", { class: "key", style: { background: "var(--series-1)", borderRadius: "50%" } }), "surprisal of sampled token"))),
            h("div", { class: "panel-body" }, trace))),
        h("div", { class: "stack" },
          h("div", { class: "panel" },
            h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "Sampling")),
            h("div", { class: "panel-body stack" },
              ctl("Temperature", temp, outs[0]), ctl("Top-p (nucleus)", topP, outs[1]), ctl("Top-k", topK, outs[2]),
              ctl("Max new tokens", maxTok, outs[3]), ctl("Repetition penalty", rep, outs[4]),
              h("div", {}, h("label", { class: "field-label" }, "Seed"), seed))),
          h("div", { class: "panel" },
            h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "Token inspector"), h("span", { class: "faint", style: { fontSize: "11px" } }, "click a token to pin")),
            inspector)))));

  emptyInspector();
  renderTrace();
  if (!ctx.status?.model) {
    anvil.replaceChildren(h("div", { class: "empty" }, h("div", {}, h("b", {}, "No checkpoint loaded"), "Train one with python -m forge.train configs/tinystories-30m.json")));
    runBtn.disabled = true;
  } else {
    run();
  }
  prompt.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); } });

  function setMode(m, btn) {
    state.mode = m;
    modeSeg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
    scaleLabel.textContent = m === "surprisal" ? "0 → 10 bits" : "0 → 6 bits";
    anvil.querySelectorAll(".tok:not(.prompt):not(.eot)").forEach((el) => { const t = state.tokens[+el.dataset.i]; el.style.setProperty("--h", tokHeat(t)); el.style.setProperty("--g", tokGlow(t)); });
  }
  // Confident tokens stay unmarked; heat (and opacity) rise with surprisal/entropy, so the
  // eye lands on the handful of places where the model actually had to choose.
  function tokHeat(t) {
    const v = Math.min(1, state.mode === "surprisal" ? bits(t.prob) / 10 : t.entropy / LN2 / 6);
    if (v < 0.2) return "transparent";
    return heat(0.35 + 0.65 * v, Math.min(1, 0.2 + v * 1.4));
  }
  // The hottest tokens (p < ~2%) also get a faint glow so they read at a glance.
  function tokGlow(t) {
    const v = Math.min(1, state.mode === "surprisal" ? bits(t.prob) / 10 : t.entropy / LN2 / 6);
    return v > 0.55 ? heat(0.4 + 0.6 * v, 0.16) : "transparent";
  }

  function emptyInspector() {
    inspector.replaceChildren(h("div", { class: "empty", style: { minHeight: "200px" } },
      h("div", {}, h("b", {}, "Nothing selected"), "Hover any generated token.")));
  }
  function inspect(t) {
    if (!t) return emptyInspector();
    const alts = t.top.some((a) => a.id === t.id) ? t.top : [...t.top.slice(0, 4), { id: t.id, text: t.text, prob: t.prob }];
    const maxp = Math.max(...alts.map((a) => a.prob));
    inspector.replaceChildren(
      h("div", { class: "inspector-token" }, `"${showToken(t.text)}"`),
      h("dl", { class: "kv" },
        h("dt", {}, "token id"), h("dd", {}, t.id),
        h("dt", {}, "position"), h("dd", {}, `+${t.i + 1}`),
        h("dt", {}, "p(token)"), h("dd", {}, fmt.pct(t.prob, 2)),
        h("dt", {}, "surprisal"), h("dd", {}, bits(t.prob).toFixed(2) + " bits"),
        h("dt", {}, "entropy"), h("dd", {}, (t.entropy / LN2).toFixed(2) + " bits"),
        h("dt", {}, "emitted at"), h("dd", {}, fmt.dur(t.t))),
      h("div", {}, h("div", { class: "field-label" }, "Top alternatives"),
        h("div", { class: "alts" }, ...alts.map((a) => h("div", { class: `alt ${a.id === t.id ? "chosen" : ""}` },
          h("span", { class: "t" }, showToken(a.text)),
          h("div", { class: "track" }, h("div", { class: "fill", style: { width: (100 * a.prob / maxp).toFixed(1) + "%" } })),
          h("span", { class: "p" }, fmt.pct(a.prob, 1)))))));
  }

  function renderTrace() {
    const toks = state.tokens;
    lineChart(trace, {
      height: 170, ariaLabel: "entropy and surprisal per generated token",
      series: [
        { name: "entropy", color: "var(--series-2)", points: toks.map((t) => ({ x: t.i, y: t.entropy / LN2 })), area: true },
        { name: "surprisal", color: "var(--series-1)", kind: "dots", r: 2.5, points: toks.map((t) => ({ x: t.i, y: bits(t.prob) })) },
      ],
      x: { label: "token", format: (v) => Math.round(v) },
      y: { min: 0, format: (v) => v + " b" },
      emptyText: "Generate to see the model's uncertainty per step",
      tooltipTitle: (x) => (toks[x] ? `#${x + 1} "${showToken(toks[x].text)}"` : `#${x}`),
      onHover: (x) => {
        anvil.querySelectorAll(".tok.pinned-h").forEach((e) => e.classList.remove("pinned-h", "pinned"));
        if (x === null || x === undefined) return;
        const el = anvil.querySelector(`.tok[data-i="${x}"]`);
        if (el) { el.classList.add("pinned", "pinned-h"); inspect(toks[x]); }
      },
    });
  }

  async function run() {
    if (state.running) state.abort?.abort();
    state.tokens = [];
    state.pinned = null;
    state.running = true;
    state.abort = new AbortController();
    runBtn.disabled = true; stopBtn.disabled = false;
    anvil.replaceChildren();
    const caret = h("span", { class: "caret" });
    anvil.append(caret);
    emptyInspector();
    let nll = 0, n = 0, lastDraw = 0;
    const t0 = performance.now();
    const body = {
      prompt: prompt.value, temperature: +temp.value, top_p: +topP.value >= 1 ? null : +topP.value,
      top_k: +topK.value || null, max_new_tokens: +maxTok.value, repetition_penalty: +rep.value,
      seed: seed.value.trim() === "" ? null : Number(seed.value),
    };
    try {
      await api.stream("/api/generate", body, (ev) => {
        if (ev.type === "prompt") {
          for (const t of ev.tokens) caret.before(h("span", { class: "tok prompt" }, t.text));
        } else if (ev.type === "token") {
          const t = { ...ev, i: state.tokens.length };
          state.tokens.push(t);
          if (ev.eot) {
            caret.before(h("span", { class: "tok eot", "data-i": t.i }, " ⏎ end of story"));
          } else {
            // leading whitespace sits outside the span so underlines hug the word itself
            const lead = t.text.match(/^\s*/)[0];
            if (lead) caret.before(lead);
            const span = h("span", { class: "tok fresh", "data-i": t.i }, t.text.slice(lead.length) || t.text);
            span.style.setProperty("--h", tokHeat(t));
            span.style.setProperty("--g", tokGlow(t));
            span.addEventListener("mouseenter", () => { if (state.pinned === null) inspect(t); });
            span.addEventListener("click", () => {
              anvil.querySelectorAll(".tok.pinned").forEach((e) => e.classList.remove("pinned"));
              if (state.pinned === t.i) { state.pinned = null; } else { state.pinned = t.i; span.classList.add("pinned"); inspect(t); }
            });
            caret.before(span);
          }
          nll += -Math.log(Math.max(t.prob, 1e-12)); n += 1;
          const dt = (performance.now() - t0) / 1000;
          document.getElementById("g-tps").innerHTML = `${(n / dt).toFixed(0)}<small>tok/s</small>`;
          document.getElementById("g-n").textContent = n;
          document.getElementById("g-bits").innerHTML = `${(nll / n / LN2).toFixed(2)}<small>bits</small>`;
          document.getElementById("g-ppl").textContent = Math.exp(nll / n).toFixed(2);
          if (n - lastDraw >= 12) { lastDraw = n; renderTrace(); }
        } else if (ev.type === "done") {
          document.getElementById("g-tps").innerHTML = `${ev.tok_per_s.toFixed(0)}<small>tok/s</small>`;
        }
      }, state.abort.signal);
    } catch (e) {
      if (e.name !== "AbortError") anvil.append(h("div", { class: "error-note" }, e.message));
    } finally {
      caret.remove();
      state.running = false;
      runBtn.disabled = false; stopBtn.disabled = true;
      renderTrace();
      const last = state.tokens.at(-1);
      if (last) inspect(state.tokens.reduce((a, b) => (bits(b.prob) > bits(a.prob) ? b : a)));
    }
  }
}
