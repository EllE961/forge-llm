// Router + shell. Each view module exports `mount(section, ctx)` and is mounted lazily the
// first time it is opened; `ctx.status` carries the /api/status payload.

import { api, fmt, h } from "./util.js";
import * as generate from "./views/generate.js";
import * as race from "./views/race.js";
import * as attention from "./views/attention.js";
import * as embeddings from "./views/embeddings.js";
import * as tokenizer from "./views/tokenizer.js";
import * as training from "./views/training.js";

const VIEWS = { generate, race, attention, embeddings, tokenizer, training };
const ORDER = Object.keys(VIEWS);
const mounted = {};
const ctx = { status: null };

function show(name) {
  if (!VIEWS[name]) name = "generate";
  for (const n of ORDER) document.getElementById(`view-${n}`).classList.toggle("active", n === name);
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === name));
  const section = document.getElementById(`view-${name}`);
  if (!mounted[name]) {
    mounted[name] = true;
    VIEWS[name].mount(section, ctx);
  } else {
    VIEWS[name].onShow?.(section, ctx);
  }
}

function renderSpec(st) {
  const spec = document.getElementById("spec");
  const m = st.model;
  const rows = m
    ? [
        ["Model", "Llama-style"],
        ["Params", fmt.compact(m.params)],
        ["Layers", `${m.config.n_layers} × d${m.config.d_model}`],
        ["Heads", `${m.config.n_heads}q / ${m.config.n_kv_heads}kv`],
        ["Trained", m.tokens_seen ? fmt.compact(m.tokens_seen) + " tok" : "—"],
        ["Val loss", fmt.fixed(m.val_loss, 3)],
        ["Draft", st.draft ? fmt.compact(st.draft.params) : "none"],
      ]
    : [["Model", "not loaded"]];
  spec.replaceChildren(
    ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]),
    h("dt", {}, "Device"),
    h("dd", {}, h("span", { class: `device-dot ${st.device === "cpu" ? "off" : ""}` }), st.device === "cpu" ? "cpu" : "cuda"),
  );
}

async function boot() {
  try {
    ctx.status = await api.get("/api/status");
    renderSpec(ctx.status);
  } catch (e) {
    ctx.status = { model: null, draft: null, device: "cpu", error: e.message };
    renderSpec(ctx.status);
  }
  addEventListener("hashchange", () => show(location.hash.slice(1)));
  addEventListener("keydown", (ev) => {
    const tag = (ev.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const i = Number(ev.key) - 1;
    if (i >= 0 && i < ORDER.length) location.hash = ORDER[i];
  });
  show(location.hash.slice(1) || "generate");
}

boot();
