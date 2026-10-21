// Speculative race: the server runs plain decoding and speculative decoding on the same prompt
// and returns per-token timestamps; both lanes are replayed side by side on that clock.

import { columnChart, lineChart } from "../charts.js";
import { api, fmt, h, readout, viewHead } from "../util.js";

const KS = [1, 2, 3, 4, 6, 8];
const TEMPS = [[0, "greedy"], [0.7, "T 0.7"], [1, "T 1.0"]];

export function mount(root, ctx) {
  const gpu = (ctx.status?.device || "").startsWith("cuda");
  // Eager is the default: it is where speculative decoding pays off at this model size;
  // the CUDA-graph engine shows the opposite regime (see README, "Inference engine").
  const st = { k: 4, temp: 0, anim: 0, engine: "eager" };
  const eSeg = h("div", { class: "seg text", title: "decode engine" },
    ...[["eager", "Eager"], ["graph", "CUDA graph"]].map(([e, l]) => h("button", {
      class: e === st.engine ? "on" : "", disabled: e === "graph" && !gpu,
      onclick: (ev) => pick(eSeg, ev.target, () => (st.engine = e)),
    }, l)));
  const prompt = h("input", { type: "text", value: "Once upon a time, there was a little girl named Lily. One day," });
  const kSeg = h("div", { class: "seg" }, ...KS.map((k) => h("button", { class: k === st.k ? "on" : "", onclick: (e) => pick(kSeg, e.target, () => (st.k = k)) }, `k=${k}`)));
  const tSeg = h("div", { class: "seg text" }, ...TEMPS.map(([t, l]) => h("button", { class: t === st.temp ? "on" : "", onclick: (e) => pick(tSeg, e.target, () => (st.temp = t)) }, l)));
  const runBtn = h("button", { class: "btn primary", onclick: () => run() }, "Run race");
  const verdict = h("div", { class: "row", style: { gap: "8px" } });

  const lane = (cls, name, sub, color) => {
    const time = h("span", { class: "lane-time" }, "—");
    const bar = h("div", { class: "run" });
    const text = h("div", { class: "anvil" });
    const el = h("div", { class: `panel lane ${cls}` },
      h("div", { class: "panel-head" },
        h("div", { class: "lane-name" }, h("span", { class: "dot", style: { background: color } }),
          h("div", {}, h("div", { class: "panel-title" }, h("b", {}, name)), h("div", { class: "faint", style: { fontSize: "11.5px" } }, sub))),
        time),
      h("div", { class: "lane-track" }, bar), text);
    return { el, time, bar, text };
  };
  const draftP = ctx.status?.draft ? fmt.compact(ctx.status.draft.params) : "?";
  const targetP = ctx.status?.model ? fmt.compact(ctx.status.model.params) : "?";
  const base = lane("base", "Autoregressive", `target ${targetP} · one forward pass per token`, "var(--ash-3)");
  const spec = lane("spec", "Speculative", `draft ${draftP} proposes k tokens · target verifies all of them in one pass`, "var(--ember)");
  const hist = h("div"), theory = h("div");

  root.append(
    viewHead("Speculative <em>race</em>",
      "Same prompt, same target model. Speculative decoding (Leviathan et al. 2023) emits tokens with exactly the target's distribution — it only changes how many target passes it takes.",
      h("div", { class: "readouts" }, readout("Speedup", "r-speed"), readout("Accept rate", "r-alpha"), readout("Tokens / target pass", "r-tpr"), readout("i.i.d. theory", "r-theory"))),
    h("div", { class: "view-body stack" },
      h("div", { class: "panel" },
        h("div", { class: "panel-body row", style: { flexWrap: "wrap" } },
          h("div", { class: "grow", style: { minWidth: "260px" } }, prompt), eSeg, kSeg, tSeg, runBtn)),
      h("div", { class: "row", style: { justifyContent: "space-between", flexWrap: "wrap" } },
        h("div", { class: "legend" },
          h("span", {}, h("i", { class: "key line", style: { background: "var(--temper)" } }), "draft token accepted"),
          h("span", {}, h("i", { class: "key line", style: { background: "var(--ember)" } }), "corrected by target"),
          h("span", {}, h("i", { class: "key line", style: { background: "var(--white-hot)" } }), "bonus token (all k accepted)"),
          h("span", {}, h("s", { style: { color: "var(--slag)", fontFamily: "var(--font-mono)" } }, "abc"), "rejected draft")),
        verdict),
      base.el, spec.el,
      h("div", { class: "grid-2" },
        h("div", { class: "panel" },
          h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "Accepted drafts per round")),
          h("div", { class: "panel-body" }, hist)),
        h("div", { class: "panel" },
          h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "Expected tokens per target pass"),
            h("div", { class: "legend" },
              h("span", {}, h("i", { class: "key line", style: { background: "var(--series-2)" } }), "(1−α^(k+1)) / (1−α)"),
              h("span", {}, h("i", { class: "key", style: { background: "var(--series-1)", borderRadius: "50%" } }), "measured"))),
          h("div", { class: "panel-body" }, theory)))));

  if (!ctx.status?.model || !ctx.status?.draft) {
    base.text.replaceChildren(h("div", { class: "empty" }, h("div", {}, h("b", {}, "Needs a target and a draft checkpoint"), "Train the draft with python -m forge.train configs/draft-6m.json")));
    runBtn.disabled = true;
    return;
  }
  drawTheory(null);
  run();

  function pick(seg, btn, set) {
    seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
    set();
  }

  function drawTheory(measured) {
    const k = st.k;
    const pts = [];
    for (let a = 0; a <= 0.995; a += 0.01) pts.push({ x: a, y: (1 - a ** (k + 1)) / (1 - a) });
    lineChart(theory, {
      height: 180,
      series: [
        { name: `k=${k}`, color: "var(--series-2)", points: pts, endDot: false },
        { name: "measured", color: "var(--series-1)", kind: "dots", r: 5, points: measured ? [measured] : [] },
      ],
      x: { label: "acceptance rate α", format: (v) => v.toFixed(1) },
      y: { min: 1, max: k + 1, format: (v) => v.toFixed(0) },
      tooltipTitle: (x) => `α = ${x.toFixed(2)}`,
    });
  }

  async function run() {
    cancelAnimationFrame(st.anim);
    runBtn.disabled = true;
    runBtn.textContent = "Running…";
    for (const l of [base, spec]) { l.text.replaceChildren(h("span", { class: "caret" })); l.bar.style.width = "0%"; l.time.textContent = "—"; l.time.classList.remove("won"); }
    verdict.replaceChildren();
    let res;
    try {
      res = await api.post("/api/race", { prompt: prompt.value, k: st.k, temperature: st.temp, max_new_tokens: 160, seed: 0,
        engine: st.engine });
    } catch (e) {
      spec.text.replaceChildren(h("div", { class: "error-note" }, e.message));
      runBtn.disabled = false; runBtn.textContent = "Run race";
      return;
    }
    runBtn.disabled = false; runBtn.textContent = "Run race";
    const s = res.stats;
    const theoryVal = s.acceptance_rate >= 1 ? st.k + 1 : (1 - s.acceptance_rate ** (st.k + 1)) / (1 - s.acceptance_rate);
    document.getElementById("r-speed").innerHTML = `${s.speedup.toFixed(2)}<small>×</small>`;
    document.getElementById("r-speed").classList.add("hot");
    document.getElementById("r-alpha").textContent = fmt.pct(s.acceptance_rate, 0);
    document.getElementById("r-tpr").textContent = s.tokens_per_round.toFixed(2);
    document.getElementById("r-theory").textContent = theoryVal.toFixed(2);
    const counts = Array(st.k + 1).fill(0);
    for (const r of res.speculative.rounds) counts[Math.min(st.k, r.n_accepted)] += 1;
    columnChart(hist, { labels: counts.map((_, i) => String(i)), values: counts, color: "var(--series-2)", height: 180,
      format: (v) => fmt.int(v), title: (l) => `${l} of ${st.k} drafts accepted` });
    drawTheory({ x: s.acceptance_rate, y: s.tokens_per_round });
    if (st.temp === 0) {
      verdict.append(s.identical
        ? h("span", { class: "chip ok" }, h("i"), "outputs identical to greedy target")
        : h("span", { class: "chip" }, "outputs differ (fp tie)"));
    } else {
      verdict.append(h("span", { class: "chip" }, "sampled: same distribution, different draw"));
    }
    if (res.engine === "graph") verdict.prepend(h("span", { class: "chip", title: "Graphed plain decoding never syncs per token, so its per-token times are interpolated from the measured total." }, "CUDA graphs · baseline timing interpolated"));
    replay(res);
  }

  function replay(res) {
    const tb = res.baseline.seconds, ts = res.speculative.seconds;
    const longest = Math.max(tb, ts, 1e-3);
    const scale = Math.max(1, 6 / longest); // slow tiny runs down so the race is watchable
    const btoks = res.baseline.tokens;
    const rounds = res.speculative.rounds;
    const nB = btoks.length, nS = rounds.reduce((a, r) => a + r.emitted.length, 0);
    let bi = 0, ri = 0;
    const bCaret = base.text.querySelector(".caret"), sCaret = spec.text.querySelector(".caret");
    const prompt = res.prompt.map((p) => p.text).join("");
    bCaret.before(h("span", { class: "tok prompt" }, prompt));
    sCaret.before(h("span", { class: "tok prompt" }, prompt));
    // Leading whitespace goes outside the span so each underline hugs its word.
    const ended = new Set();  // lanes stop rendering after the end-of-story token
    const put = (caret, text, cls) => {
      if (ended.has(caret)) return;
      if (text === "<|endoftext|>") { caret.before(h("span", { class: "tok eot" }, " ⏎ end of story")); ended.add(caret); return; }
      const lead = text.match(/^\s*/)[0];
      if (lead) caret.before(lead);
      caret.before(h("span", { class: `tok fresh ${cls}` }, text.slice(lead.length) || text));
    };
    const passes = (n, extra) => h("span", { class: "faint" }, `  ·  ${n} target passes${extra || ""}`);
    const t0 = performance.now();
    const tick = () => {
      const now = (performance.now() - t0) / 1000 / scale;
      while (bi < nB && btoks[bi].t <= now) put(bCaret, btoks[bi++].text, "");
      while (ri < rounds.length && rounds[ri].t <= now) {
        const r = rounds[ri++];
        for (const g of r.rejected) sCaret.before(h("span", { class: "rejected" }, g.text));
        for (const e of r.emitted) put(sCaret, e.text, e.kind === "accepted" ? "acc" : e.kind === "bonus" ? "bon" : "res");
      }
      base.bar.style.width = (100 * Math.min(1, now / tb)).toFixed(1) + "%";
      spec.bar.style.width = (100 * Math.min(1, now / ts)).toFixed(1) + "%";
      if (bi >= nB) base.time.replaceChildren(`${fmt.dur(tb)} · ${(nB / tb).toFixed(0)} tok/s`, passes(nB));
      else base.time.textContent = fmt.dur(Math.min(now, tb));
      if (ri >= rounds.length) {
        const drafted = rounds.reduce((a, r) => a + r.drafted, 0);
        spec.time.replaceChildren(`${fmt.dur(ts)} · ${(nS / ts).toFixed(0)} tok/s`, passes(rounds.length, ` + ${drafted} draft`));
      } else spec.time.textContent = fmt.dur(Math.min(now, ts));
      if (bi >= nB && ri >= rounds.length) {
        bCaret.remove(); sCaret.remove();
        (ts <= tb ? spec : base).time.classList.add("won");
        spec.text.querySelectorAll(".rejected").forEach((g) => g.remove());
        return;
      }
      st.anim = requestAnimationFrame(tick);
    };
    st.anim = requestAnimationFrame(tick);
  }
}
