// Tokenizer: live BPE segmentation with our 8k vocabulary vs. GPT-4's cl100k (100k).

import { api, debounce, fmt, h, readout, tokenNodes, viewHead } from "../util.js";

const SAMPLES = {
  Story: "Once upon a time, there was a little girl named Lily. She loved to play outside with her dog, Max. One day, they found a big, shiny rock.",
  Code: "def train_step(model, batch):\n    logits = model(batch.x)\n    loss = F.cross_entropy(logits.view(-1, V), batch.y.view(-1))\n    return loss",
  Multilingual: "Il était une fois une petite fille. Érase una vez una niña. 昔々、小さな女の子がいました。",
  Numbers: "In 2024 the model saw 141,664,307 tokens at 4.24 bytes/token — about 600 MB of text.",
};

export function mount(root) {
  const input = h("textarea", { rows: 5, spellcheck: "false" }, SAMPLES.Story);
  const ours = h("div", { class: "tok-out" });
  const theirs = h("div", { class: "tok-out" });
  const bars = h("div", { class: "stack" });
  const samples = h("div", { class: "seg text" }, ...Object.keys(SAMPLES).map((k, i) =>
    h("button", { class: i === 0 ? "on" : "", onclick: (e) => { samples.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === e.target)); input.value = SAMPLES[k]; run(); } }, k)));
  const tip = h("div", { class: "muted", style: { fontSize: "12px", lineHeight: 1.6 } });

  root.append(
    viewHead("Tokenizer <em>byte-level BPE</em>",
      "Trained from scratch on 100 MB of TinyStories in ~5 s (incremental pair counts + lazy max-heap). Same GPT-4 pre-split regex as cl100k, so the comparison isolates the learned merges.",
      h("div", { class: "readouts" }, readout("Tokens", "t-n"), readout("Bytes / token", "t-bpt"), readout("cl100k tokens", "t-cl"), readout("vs cl100k", "t-ratio"))),
    h("div", { class: "view-body" },
      h("div", { class: "split" },
        h("div", { class: "stack" },
          h("div", { class: "panel" }, h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "Input"), samples), h("div", { class: "panel-body" }, input)),
          h("div", { class: "panel" }, h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, h("b", {}, "Forge BPE"), " · 8,192 tokens · trained on stories"), h("span", { class: "faint", id: "t-ours-n" })), ours),
          h("div", { class: "panel" }, h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, h("b", {}, "cl100k_base"), " · 100,277 tokens · GPT-4"), h("span", { class: "faint", id: "t-cl-n" })), theirs)),
        h("div", { class: "stack" },
          h("div", { class: "panel" }, h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "Compression")), h("div", { class: "panel-body stack" }, bars)),
          h("div", { class: "panel" }, h("div", { class: "panel-head" }, h("div", { class: "panel-title" }, "Reading the result")), h("div", { class: "panel-body" }, tip))))));

  input.addEventListener("input", debounce(run, 160));
  run();

  function chips(el, toks) {
    el.replaceChildren(...toks.map((t) => h("span", { class: "tk", title: `id ${t.id}${t.bytes ? ` · ${t.bytes} bytes` : ""}` }, ...tokenNodes(t.text, true))));
  }

  async function run() {
    try {
      const r = await api.post("/api/tokenize", { text: input.value });
      chips(ours, r.tokens);
      document.getElementById("t-n").textContent = fmt.int(r.n_tokens);
      document.getElementById("t-bpt").textContent = r.bytes_per_token.toFixed(2);
      document.getElementById("t-ours-n").textContent = `${r.n_tokens} tokens`;
      const rows = [["Forge 8k", r.bytes_per_token, "var(--series-1)", r.n_tokens]];
      if (r.cl100k) {
        chips(theirs, r.cl100k.tokens);
        document.getElementById("t-cl").textContent = fmt.int(r.cl100k.n_tokens);
        const ratio = r.n_tokens / Math.max(1, r.cl100k.n_tokens);
        document.getElementById("t-ratio").innerHTML = `${ratio.toFixed(2)}<small>× tokens</small>`;
        document.getElementById("t-cl-n").textContent = `${r.cl100k.n_tokens} tokens`;
        rows.push(["cl100k 100k", r.cl100k.bytes_per_token, "var(--series-2)", r.cl100k.n_tokens]);
        tip.textContent = ratio <= 1.15
          ? "On text like the training distribution, a vocabulary 12× smaller compresses almost as well — every merge was spent on words that actually occur in children's stories."
          : "Off-distribution text (code, other scripts, numbers) falls apart into short pieces: those byte sequences were never frequent enough in TinyStories to earn a merge. This is why tokenizers are trained on the same mixture as the model.";
      } else {
        theirs.replaceChildren(h("div", { class: "empty" }, "tiktoken not installed"));
      }
      const max = Math.max(...rows.map((x) => x[1]));
      bars.replaceChildren(
        h("div", { class: "field-label" }, "bytes per token (higher = better compression)"),
        ...rows.map(([name, v, col, n]) => h("div", { class: "compare-bar" },
          h("span", { class: "muted" }, name),
          h("div", { class: "track" }, h("div", { class: "fill", style: { width: (100 * v / max).toFixed(1) + "%", background: col } })),
          h("span", { class: "mono", style: { textAlign: "right" } }, v.toFixed(2)))),
        h("div", { class: "faint", style: { fontSize: "11.5px" } }, `${r.n_bytes} bytes · ${r.n_chars} characters · whitespace shown as ·`),
      );
    } catch (e) {
      ours.replaceChildren(h("div", { class: "error-note" }, e.message));
    }
  }
}
