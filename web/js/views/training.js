// Training runs: curves straight from runs/<name>/metrics.jsonl, refreshed live while a run
// is still writing (no summary.json yet).

import { ema, lineChart } from "../charts.js";
import { api, fmt, h, readout, viewHead } from "../util.js";

const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];

export function mount(root) {
  const st = { runs: [], name: null, timer: 0 };
  const seg = h("div", { class: "seg text" });
  const live = h("span");
  const charts = { loss: h("div"), lr: h("div"), gn: h("div"), tps: h("div") };
  const samples = h("div", { class: "panel-body stack" });
  const compare = h("div");
  const comparePanel = h("div", { class: "panel", style: { display: "none" } },
    h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "Fine-tuning on TinyShakespeare · validation loss"), h("div", { class: "legend", id: "cmp-legend" })),
    h("div", { class: "panel-body" }, compare));

  const card = (title, legend, el) => h("div", { class: "panel" },
    h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, title), legend || h("span")), h("div", { class: "panel-body" }, el));
  root.append(
    viewHead("Training <em>runs</em>",
      "Every curve is read from the JSONL the trainer writes. Loss is plotted against tokens seen; the schedule is warmup + cosine, budgeted on wall-clock time for the main run.",
      h("div", { class: "readouts" }, readout("Best val loss", "tr-val"), readout("Perplexity", "tr-ppl"), readout("Tokens seen", "tr-tok"),
        readout("Wall-clock", "tr-time"), readout("Median MFU", "tr-mfu"))),
    h("div", { class: "view-body stack" },
      h("div", { class: "row", style: { justifyContent: "space-between" } }, seg, live),
      h("div", { class: "grid-2" },
        card("Loss", h("div", { class: "legend" },
          h("span", {}, h("i", { class: "key line", style: { background: "var(--series-1)" } }), "train (EMA)"),
          h("span", {}, h("i", { class: "key", style: { background: "var(--series-2)", borderRadius: "50%" } }), "validation")), charts.loss),
        card("Learning rate", null, charts.lr),
        card("Gradient norm (pre-clip, EMA)", null, charts.gn),
        card("Throughput", null, charts.tps)),
      comparePanel,
      h("div", { class: "panel" }, h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "Latest samples")), samples)));

  refreshRuns();

  async function refreshRuns() {
    try { st.runs = await api.get("/api/runs"); } catch { st.runs = []; }
    if (!st.runs.length) {
      charts.loss.replaceChildren(h("div", { class: "empty" }, h("div", {}, h("b", {}, "No runs yet"), "python -m forge.train configs/tinystories-30m.json")));
      return;
    }
    const order = (n) => (n.startsWith("tinystories") ? 0 : n.startsWith("draft") ? 1 : 2);
    st.runs.sort((a, b) => order(a.name) - order(b.name) || a.name.localeCompare(b.name));
    if (!st.name) st.name = st.runs[0].name;
    seg.replaceChildren(...st.runs.map((r) => h("button", { class: r.name === st.name ? "on" : "", onclick: () => { st.name = r.name; refreshRuns(); } }, r.name)));
    load();
    drawCompare();
  }

  async function load() {
    clearTimeout(st.timer);
    const d = await api.get(`/api/runs/${encodeURIComponent(st.name)}`);
    const active = !d.summary;
    live.replaceChildren(active ? h("span", { class: "chip live" }, h("i"), "live · refreshing every 5 s") : h("span", { class: "chip" }, "finished"));
    const tr = d.train, ev = d.eval;
    const best = ev.length ? Math.min(...ev.map((e) => e.val_loss)) : null;
    const last = tr.at(-1);
    const mfu = tr.map((r) => r.mfu).sort((a, b) => a - b);
    const med = (a) => (a.length ? a[Math.floor(a.length / 2)] : null);
    document.getElementById("tr-val").textContent = fmt.fixed(best, 3);
    document.getElementById("tr-ppl").textContent = best ? Math.exp(best).toFixed(2) : "—";
    document.getElementById("tr-tok").textContent = last ? fmt.compact(last.tokens) : "—";
    document.getElementById("tr-time").textContent = last ? fmt.dur(last.elapsed) : "—";
    document.getElementById("tr-mfu").textContent = med(mfu) ? fmt.pct(med(mfu), 1) : "—";

    const xTok = { label: "tokens", format: (v) => fmt.compact(v) };
    const pts = (key) => tr.filter((r) => r.step > 1).map((r) => ({ x: r.tokens, y: r[key] }));
    lineChart(charts.loss, {
      height: 230,
      series: [
        { name: "train (raw)", color: "var(--series-1)", points: pts("loss"), dim: true, width: 1, endDot: false },
        { name: "train (EMA)", color: "var(--series-1)", points: ema(pts("loss"), 0.6) },
        { name: "validation", color: "var(--series-2)", kind: "dots", r: 3.5, points: ev.filter((e) => e.step > 0).map((e) => ({ x: e.tokens, y: e.val_loss })) },
      ],
      x: xTok, y: { log: true, format: (v) => v.toFixed(v < 2 ? 2 : 1) },
      tooltipTitle: (x) => `${fmt.compact(x)} tokens`,
    });
    lineChart(charts.lr, { height: 170, series: [{ name: "lr", color: "var(--series-3)", points: pts("lr") }], x: xTok, y: { min: 0, format: (v) => (v === 0 ? "0" : v.toExponential(1)) }, tooltipTitle: (x) => `${fmt.compact(x)} tokens` });
    lineChart(charts.gn, { height: 170, series: [{ name: "grad norm", color: "var(--series-3)", points: ema(pts("grad_norm"), 0.7) }], x: xTok, y: { min: 0, format: (v) => v.toFixed(1) }, tooltipTitle: (x) => `${fmt.compact(x)} tokens` });
    lineChart(charts.tps, {
      height: 170, series: [{ name: "tokens/s", color: "var(--series-3)", points: tr.filter((r) => r.step > 1).map((r) => ({ x: r.elapsed / 60, y: r.tok_per_s })), format: (v) => fmt.compact(v) }],
      x: { label: "minutes", format: (v) => v.toFixed(0) }, y: { min: 0, format: (v) => fmt.compact(v) }, tooltipTitle: (x) => `${x.toFixed(1)} min`,
    });

    const s = d.samples.at(-1);
    samples.replaceChildren(...(s ? s.samples.map((x) => h("div", { class: "sample" }, h("b", {}, x.prompt), x.completion)) : [h("div", { class: "muted" }, "No samples logged yet.")]));
    if (s) samples.prepend(h("div", { class: "faint", style: { fontSize: "11.5px" } }, `step ${fmt.int(s.step)} · temperature 0.8 · top-p 0.95`));
    if (active) st.timer = setTimeout(load, 5000);
  }

  async function drawCompare() {
    const ft = st.runs.filter((r) => r.name.startsWith("shakespeare-"));
    if (ft.length < 2) return;
    comparePanel.style.display = "";
    const ds = await Promise.all(ft.slice(0, 3).map((r) => api.get(`/api/runs/${encodeURIComponent(r.name)}`)));
    document.getElementById("cmp-legend").replaceChildren(...ds.map((d, i) =>
      h("span", {}, h("i", { class: "key line", style: { background: SERIES[i] } }), `${d.name.replace("shakespeare-", "")} · ${fmt.compact(d.summary?.n_trainable)} trainable`)));
    lineChart(compare, {
      height: 230,
      series: ds.map((d, i) => ({ name: d.name.replace("shakespeare-", ""), color: SERIES[i], points: d.eval.map((e) => ({ x: e.step, y: e.val_loss })) })),
      x: { label: "step", format: (v) => Math.round(v) }, y: { format: (v) => v.toFixed(2) }, tooltipTitle: (x) => `step ${x}`,
    });
  }
}
