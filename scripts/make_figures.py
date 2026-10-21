"""Render the README figures from runs/*/metrics.jsonl and results/*.json.

Every figure shares one style: soot background, hairline solid grid, 2px lines, a single
y-axis per panel, legends only when there are two or more series, and the playground's
validated three-colour palette (ember, temper blue, temper purple) plus its heat ramp.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere without installing
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = Path("docs/assets")
RUNS = Path("runs")
RES = Path("results")

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE = "#d95926", "#3987e5", "#c0509c"
HEAT = LinearSegmentedColormap.from_list("heat", ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c", "#d9520f",
                                                  "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"])

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "xtick.major.pad": 6, "ytick.major.pad": 6, "axes.axisbelow": True,
    "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def ema(y: np.ndarray, a: float) -> np.ndarray:
    out, acc = np.empty_like(y), 0.0
    for i, v in enumerate(y):
        acc = a * acc + (1 - a) * v
        out[i] = acc / (1 - a ** (i + 1))
    return out


def human(n: float) -> str:
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.0f}{suf}" if n / div >= 10 else f"{n / div:.1f}{suf}"
    return f"{n:.0f}"


def finish(fig, name: str, note: str | None = None) -> None:
    if note:
        fig.text(0.012, -0.015, note, color=INK3, fontsize=8.5, ha="left", va="top")  # below the axes
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[fig] {OUT / name}.png")


def end_label(ax, x, y, text, color=INK2, dx=6, dy=0) -> None:
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=color, fontsize=9, va="center")


# --------------------------------------------------------------------------- figures


def fig_pretraining() -> None:
    main = load_jsonl(RUNS / "tinystories-30m" / "metrics.jsonl")
    if not main:
        return
    draft = load_jsonl(RUNS / "draft-6m" / "metrics.jsonl")
    tr = [r for r in main if r["type"] == "train" and r["step"] > 1]
    ev = [r for r in main if r["type"] == "eval" and r["step"] > 0]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.array([r["tokens"] for r in tr]) / 1e6
    y = np.array([r["loss"] for r in tr])
    ax.plot(x, y, color=EMBER, alpha=0.22, linewidth=1)
    ax.plot(x, ema(y, 0.6), color=EMBER, label="27M · train")
    ax.plot([r["tokens"] / 1e6 for r in ev], [r["val_loss"] for r in ev], "o", color=TEMPER, markersize=5,
            markeredgecolor=PANEL, markeredgewidth=1.5, label="27M · validation")
    if draft:
        dev = [r for r in draft if r["type"] == "eval" and r["step"] > 0]
        ax.plot([r["tokens"] / 1e6 for r in dev], [r["val_loss"] for r in dev], "-", color=PURPLE, linewidth=1.6,
                label="6M draft · validation")
    ax.set_yscale("log")
    ax.set_yticks([1, 1.25, 1.5, 2, 3, 4, 6])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3g"))
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(min(r["val_loss"] for r in ev) * 0.88, min(7.5, max(y) * 1.05))
    ax.set_xlabel("tokens seen (millions)")
    ax.set_ylabel("cross-entropy (nats / token)")
    best = min(ev, key=lambda r: r["val_loss"])
    ax.annotate(f"best val {best['val_loss']:.3f}  ·  ppl {math.exp(best['val_loss']):.2f}",
                (best["tokens"] / 1e6, best["val_loss"]), xytext=(-12, 26), textcoords="offset points", ha="right",
                color=INK, fontsize=9.5, arrowprops={"arrowstyle": "-", "color": INK3, "lw": 0.8})
    ax.set_title("Pretraining on TinyStories")
    ax.legend(loc="upper right")
    mins = tr[-1]["elapsed"] / 60
    finish(fig, "loss_curves", f"75-minute wall-clock budget on a shared RTX 3080 · {human(tr[-1]['tokens'])} tokens · "
                               f"{mins:.0f} min")


def fig_dynamics() -> None:
    main = load_jsonl(RUNS / "tinystories-30m" / "metrics.jsonl")
    tr = [r for r in main if r["type"] == "train" and r["step"] > 1]
    if not tr:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    t = np.array([r["elapsed"] for r in tr]) / 60
    axes[0].plot(t, [r["lr"] * 1e3 for r in tr], color=PURPLE)
    axes[0].set_title("Learning rate (×10⁻³)")
    axes[0].set_xlabel("minutes")
    axes[1].plot(t, ema(np.array([r["tok_per_s"] for r in tr]) / 1e3, 0.6), color=PURPLE)
    axes[1].set_title("Throughput (K tokens/s)")
    axes[1].set_xlabel("minutes")
    axes[1].set_ylim(bottom=0)
    gn = ema(np.array([r["grad_norm"] for r in tr]), 0.7)
    axes[2].plot(np.array([r["tokens"] for r in tr]) / 1e6, gn, color=PURPLE)
    axes[2].set_title("Gradient norm (pre-clip)")
    axes[2].set_xlabel("tokens (millions)")
    axes[2].set_ylim(bottom=0)
    fig.tight_layout(w_pad=3)
    finish(fig, "training_dynamics", "Cosine schedule driven by elapsed time (Cramming-style budget); "
                                     "throughput varies because the GPU was shared with other jobs.")


def fig_kv_cache() -> None:
    for dev in ("cuda", "cpu"):
        r = load_json(RES / f"kv_cache_{dev}.json")
        if not r:
            continue
        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        n = len(r["cache"]["latency_ms"])
        pos = np.arange(n) + r["meta"]["prompt_len"] + 1
        ax.plot(pos, ema(np.array(r["no_cache"]["latency_ms"]), 0.8), color=EMBER)
        ax.plot(pos, ema(np.array(r["cache"]["latency_ms"]), 0.8), color=TEMPER)
        end_label(ax, pos[-1], ema(np.array(r["no_cache"]["latency_ms"]), 0.8)[-1],
                  f"recompute · {r['no_cache']['tok_per_s']:.0f} tok/s", INK2)
        end_label(ax, pos[-1], ema(np.array(r["cache"]["latency_ms"]), 0.8)[-1],
                  f"KV cache · {r['cache']['tok_per_s']:.0f} tok/s", INK2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("sequence position")
        ax.set_ylabel("ms per generated token")
        ax.set_title(f"Decode latency vs. position ({dev.upper()}) · KV cache {r['speedup']:.1f}× faster")
        ax.spines["bottom"].set_color(GRID)
        finish(fig, f"kv_cache_{dev}", f"GQA (2 KV heads for 8 query heads) keeps the cache at "
                                       f"{r['cache_mb_per_sequence']:.1f} MB/sequence vs "
                                       f"{r['mha_cache_mb_per_sequence']:.1f} MB for MHA (fp32, 512 positions).")


def fig_speculative() -> None:
    for dev in ("cuda", "cpu"):
        r = load_json(RES / f"speculative_{dev}.json")
        if not r:
            continue
        rows = r["rows"]
        engines = [e for e in ("eager", "graph") if any(x.get("engine", "eager") == e for x in rows)]
        temps = sorted({x["temperature"] for x in rows})
        colors = [EMBER, TEMPER, PURPLE]
        fig, axes = plt.subplots(1, 1 + len(engines), figsize=(5.4 * (1 + len(engines)), 3.8))
        opts = {"markersize": 5, "markeredgecolor": PANEL, "markeredgewidth": 1.5}
        for i, tmp in enumerate(temps):
            lab = "greedy" if tmp == 0 else f"T = {tmp}"
            for j, eng in enumerate(engines):
                rr = sorted([x for x in rows if x["temperature"] == tmp and x.get("engine", "eager") == eng],
                            key=lambda x: x["k"])
                ks = [x["k"] for x in rr]
                if j == 0:
                    axes[0].plot(ks, [x["tokens_per_round"] for x in rr], "-o", color=colors[i], label=lab, **opts)
                ax = axes[1 + j]
                ax.plot(ks, [x["speedup"] for x in rr], "-o", color=colors[i], label=lab, **opts)
                if tmp == 0:  # label only the greedy optimum to avoid collisions
                    best = max(rr, key=lambda x: x["speedup"])
                    end_label(ax, best["k"], best["speedup"], f"{best['speedup']:.2f}×", INK, dx=-4, dy=11)
        axes[0].set_title("Tokens per target forward pass")
        axes[0].set_ylim(bottom=1)
        titles = {"eager": "Speedup · eager PyTorch", "graph": "Speedup · CUDA-graph engine"}
        for j, eng in enumerate(engines):
            axes[1 + j].axhline(1, color=INK3, linewidth=1)
            axes[1 + j].set_title(titles[eng] + " (vs. same-engine plain decoding)", fontsize=11)
        for ax in axes:
            ax.set_xlabel("draft length k")
            ax.set_xticks(sorted({x["k"] for x in rows}))
        axes[0].legend(loc="upper left")
        fig.tight_layout(w_pad=3)
        m = r["meta"]
        shared = " · GPU shared with other training jobs during measurement" if dev == "cuda" else ""
        finish(fig, f"speculative_{dev}", f"{dev.upper()} · target {human(m['target_params'])} · draft "
                                          f"{human(m['draft_params'])} · {m['prompts']} held-out prompts × "
                                          f"{m['tokens_per_prompt']} tokens{shared}")


def fig_engines() -> None:
    """One bar per decoding strategy, same model, same GPU, greedy, tokens/s."""
    kv = load_json(RES / "kv_cache_cuda.json")
    sp = load_json(RES / "speculative_cuda.json")
    if not kv:
        return
    bars = [("no KV cache\n(recompute prefix)", kv["no_cache"]["tok_per_s"]),
            ("KV cache\n(eager PyTorch)", kv["cache"]["tok_per_s"])]
    if "graph" in kv:
        bars.append(("KV cache\n+ CUDA graph", kv["graph"]["tok_per_s"]))
    if sp:
        g = [x for x in sp["rows"] if x.get("engine") == "graph" and x["temperature"] == 0]
        if g:
            best = max(g, key=lambda x: x["tok_per_s"])
            bars.append((f"CUDA graph\n+ speculative (k={best['k']})", best["tok_per_s"]))
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    xs = np.arange(len(bars))
    vals = [b[1] for b in bars]
    ax.bar(xs, vals, width=0.5, color=[EMBER if i == int(np.argmax(vals)) else "#5a524b" for i in range(len(bars))])
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:,.0f}", (x, v), xytext=(0, 5), textcoords="offset points", ha="center", color=INK, fontsize=10)
    ax.set_xticks(xs, [b[0] for b in bars])
    ax.tick_params(axis="x", colors=INK2, labelsize=9.5)
    ax.set_ylabel("tokens / s (greedy, batch 1)")
    ax.set_title("Decoding engines on the same 27M model")
    finish(fig, "engines", "RTX 3080 shared with other training jobs during measurement; prompt 16 tokens, "
                           f"{kv['meta']['new_tokens']} new tokens (speculative: 8 prompts × 128 tokens).")


def fig_lora() -> None:
    runs = [("shakespeare-lora-r4", "LoRA r=4", EMBER), ("shakespeare-lora-r16", "LoRA r=16", TEMPER),
            ("shakespeare-full-ft", "full fine-tune", PURPLE)]
    data = [(name, lab, c, load_jsonl(RUNS / name / "metrics.jsonl"), load_json(RUNS / name / "summary.json"))
            for name, lab, c in runs]
    data = [d for d in data if d[3]]
    if len(data) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for _, lab, c, recs, _ in data:
        ev = [r for r in recs if r["type"] == "eval"]
        ax.plot([r["step"] for r in ev], [r["val_loss"] for r in ev], color=c, label=lab)
    base = [r for r in data[0][3] if r["type"] == "eval"][0]["val_loss"]
    ax.axhline(base, color=INK3, linewidth=1)
    end_label(ax, 0, base, f"TinyStories model, no fine-tuning · {base:.2f}", INK3, dx=4, dy=8)
    ax.set_title("Validation loss on held-out Shakespeare")
    ax.set_xlabel("step")
    ax.set_ylabel("nats / token")
    ax.legend(loc="center right")
    ax2 = axes[1]  # dot plot on a log axis: no bar baseline to misread
    labels = [d[1] for d in data]
    vals = [d[4]["n_trainable"] if d[4] else 0 for d in data]
    total = data[-1][4]["n_params"] if data[-1][4] else max(vals)
    ys = np.arange(len(vals))
    ax2.hlines(ys, 1e5, vals, color=GRID, linewidth=1)
    ax2.scatter(vals, ys, s=70, c=[d[2] for d in data], edgecolors=PANEL, linewidths=2, zorder=3)
    ax2.set_yticks(ys, labels)
    ax2.set_xscale("log")
    ax2.set_xlim(1e5, 1e8)
    ax2.set_ylim(len(vals) - 0.5, -0.5)
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        text = f"{human(v)} · {100 * v / total:.1f}%" if v < total else f"{human(v)} · all"
        ax2.annotate(text, (v, i), xytext=(10, 0), textcoords="offset points",
                     va="center", color=INK, fontsize=9)
    ax2.set_title("Trainable parameters")
    ax2.tick_params(axis="y", colors=INK2, labelsize=9.5)
    fig.tight_layout(w_pad=3)
    finish(fig, "lora", "Same base checkpoint, steps, batch and data for every run; LoRA on all attention + MLP "
                        "projections.")


def fig_attention_memory() -> None:
    r = load_json(RES / "attention_memory.json")
    if not r:
        return
    rows = r["rows"]
    bss = sorted({x["batch_size"] for x in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    w = 0.36
    for j, (key, title, fmt) in enumerate([("peak_mem_gb", "Peak training memory (GB)", "{:.2f}"),
                                           ("tok_per_s", "Throughput (K tokens/s)", "{:.0f}")]):
        ax = axes[j]
        for i, (impl, c) in enumerate([("naive", EMBER), ("chunked", TEMPER)]):
            vals = [next(x[key] for x in rows if x["batch_size"] == b and x["attention"] == impl) for b in bss]
            vals = [v / 1e3 for v in vals] if key == "tok_per_s" else vals
            xs = np.arange(len(bss)) + (i - 0.5) * w
            ax.bar(xs, vals, width=w - 0.04, color=c, label=impl)
            for xx, v in zip(xs, vals):
                ax.annotate(fmt.format(v), (xx, v), xytext=(0, 4), textcoords="offset points", ha="center", color=INK,
                            fontsize=9)
        ax.set_xticks(range(len(bss)), [f"micro-batch {b}" for b in bss])
        ax.set_title(title)
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=3)
    finish(fig, "attention_memory", "27M model, T = 512, bf16 autocast, torch 1.13 (no fused SDPA). Chunked = custom "
                                    "autograd function that recomputes attention probabilities in backward.")


def fig_heads() -> None:
    """Attention patterns of characteristic heads on a repeated random-token probe."""
    ckpt = Path("checkpoints/tinystories-30m/model.safetensors")
    if not ckpt.exists():
        return
    import torch

    from forge.model import Transformer

    model = Transformer.from_checkpoint(ckpt).eval()
    g = torch.Generator().manual_seed(0)
    L = 24
    rand = torch.randint(300, 8000, (1, L), generator=g)
    seq = torch.cat([rand, rand], 1)
    with torch.no_grad():
        model(seq, capture_attn=True)
    maps = [a[0] for a in model.attention_maps()]  # (H, 2L, 2L)
    idx = torch.arange(L, 2 * L)
    scores = []
    for li, a in enumerate(maps):
        prev = a[:, torch.arange(1, 2 * L), torch.arange(0, 2 * L - 1)].mean(-1)
        ind = a[:, idx, idx - L + 1].mean(-1)
        sink = a[:, 1:, 0].mean(-1)
        for h in range(a.shape[0]):
            scores.append((li, h, float(prev[h]), float(ind[h]), float(sink[h])))
    picks = [("strongest previous-token head", max(scores, key=lambda s: s[2])),
             ("most induction-like head", max(scores, key=lambda s: s[3])),
             ("most first-token-focused head", max(scores, key=lambda s: s[4]))]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1))
    for ax, (name, (li, h, p, ind, sk)) in zip(axes, picks):
        m = maps[li][h].numpy()
        ax.imshow(np.sqrt(m), cmap=HEAT, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{name}\nL{li} · H{h}", fontsize=11)
        ax.set_xticks([0, L, 2 * L - 1], ["0", f"{L}", f"{2 * L - 1}"])
        ax.set_yticks([0, L, 2 * L - 1], ["0", f"{L}", f"{2 * L - 1}"])
        ax.grid(False)
        ax.axhline(L - 0.5, color=INK3, linewidth=0.6)
        ax.axvline(L - 0.5, color=INK3, linewidth=0.6)
        ax.set_xlabel(f"prev {p:.2f} · induction {ind:.2f} · sink {sk:.2f}", fontsize=9)
    fig.tight_layout(w_pad=2)
    finish(fig, "attention_heads", f"Probe: {L} random tokens repeated twice; colour = √attention. "
                                   "A true induction head would light "
                                   "the offset diagonal in the lower-left block (none clearly did at this scale).")


def fig_tokenizer() -> None:
    tok_path = Path("data/tokenizer.json")
    if not tok_path.exists():
        return
    try:
        import tiktoken
    except ImportError:
        return
    from forge.tokenizer import Tokenizer

    ours = Tokenizer.load(tok_path)
    cl = tiktoken.get_encoding("cl100k_base")
    valid = Path("data/tinystories/valid.txt").read_text(encoding="utf-8")[:300_000].replace("<|endoftext|>", "\n")
    code = "".join(p.read_text(encoding="utf-8") for p in sorted(Path("forge").glob("*.py")))[:300_000]
    multi = ("Il était une fois une petite fille qui aimait jouer dehors. Érase una vez un niño que tenía un perro. "
             "Es war einmal ein kleiner Junge. 昔々、小さな女の子がいました。Жила-была маленькая девочка. ") * 40
    texts = [("TinyStories (val)", valid), ("Python source", code), ("Multilingual", multi)]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    w = 0.3
    for i, (name, enc, c) in enumerate([("Forge BPE · 8,192", ours.encode_ordinary, EMBER),
                                        ("cl100k · 100,256", cl.encode_ordinary, TEMPER)]):
        vals = [len(t.encode("utf-8")) / len(enc(t)) for _, t in texts]
        xs = np.arange(len(texts)) + (i - 0.5) * w
        ax.bar(xs, vals, width=w - 0.04, color=c, label=name)
        for xx, v in zip(xs, vals):
            ax.annotate(f"{v:.2f}", (xx, v), xytext=(0, 4), textcoords="offset points", ha="center", color=INK,
                        fontsize=9)
    ax.set_xticks(range(len(texts)), [n for n, _ in texts])
    ax.set_title("Compression: UTF-8 bytes per token (higher is better)")
    ax.legend(loc="upper right")
    finish(fig, "tokenizer", "Same GPT-4 pre-tokenisation regex for both; only the learned merges differ.")


def main() -> None:
    for f in (fig_pretraining, fig_dynamics, fig_kv_cache, fig_speculative, fig_engines, fig_lora, fig_attention_memory,
              fig_heads, fig_tokenizer):
        try:
            f()
        except Exception as e:  # keep going: each figure is independent
            print(f"[fig] {f.__name__} failed: {e!r}")


if __name__ == "__main__":
    main()
