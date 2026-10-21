"""Playground backend (FastAPI).

Serves the single-page UI in ``web/`` and a small JSON/SSE API over the trained models:

    GET  /api/status            model + draft metadata
    POST /api/generate          SSE: one event per token (prob, entropy, top-5 alternatives)
    POST /api/race              speculative vs. plain decoding on the same prompt, with timings
    POST /api/attention         per-layer/per-head attention maps for a prompt (uint8, base64)
    GET  /api/heads             per-head summary scores (previous-token, induction, sink, entropy)
    POST /api/tokenize          token spans for our BPE and (if available) cl100k
    GET  /api/runs[/{name}]     training metrics written by forge.train
    GET  /api/embeddings        2-D PCA of the token embedding matrix
    GET  /api/neighbors/{id}    nearest tokens by cosine similarity in embedding space

Run::

    python -m forge.server --ckpt checkpoints/tinystories-30m/best.pt \
        --draft checkpoints/draft-6m/best.pt --port 8001
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from forge.fastpath import StaticDecoder, fast_generate, fast_speculative
from forge.generate import stream
from forge.model import Transformer
from forge.speculative import speculative_stream
from forge.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
# Windows registries often map .js to text/plain, which browsers refuse for ES modules.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


# --------------------------------------------------------------------------- request models


class GenerateRequest(BaseModel):
    prompt: str = "Once upon a time"
    max_new_tokens: int = Field(200, ge=1, le=1024)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(None, ge=1)
    top_p: float | None = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)
    seed: int | None = None


class RaceRequest(BaseModel):
    prompt: str = "Once upon a time"
    max_new_tokens: int = Field(160, ge=1, le=400)
    k: int = Field(4, ge=1, le=12)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    seed: int | None = 0
    engine: Literal["eager", "graph"] = "eager"


class TextRequest(BaseModel):
    text: str = Field("", max_length=20000)


# --------------------------------------------------------------------------- state


class Playground:
    """Holds models + tokenizer; a lock serialises GPU work across request threads."""

    def __init__(self, model: Transformer | None, tokenizer: Tokenizer, draft: Transformer | None = None,
                 runs_dir: Path = ROOT / "runs", meta: dict[str, Any] | None = None):
        self.model = model.eval() if model is not None else None
        self.draft = draft.eval() if draft is not None else None
        self.tok = tokenizer
        self.runs_dir = Path(runs_dir)
        self.meta = meta or {}
        self.lock = threading.Lock()
        self._heads: dict[str, Any] | None = None
        self._emb: dict[str, Any] | None = None
        self._cl100k = None

    @classmethod
    def from_paths(cls, ckpt: str | None, tokenizer: str, draft: str | None, device: str,
                   runs_dir: Path) -> Playground:
        tok = Tokenizer.load(tokenizer)
        model = Transformer.from_checkpoint(ckpt, device) if ckpt and Path(ckpt).exists() else None
        dmodel = Transformer.from_checkpoint(draft, device) if draft and Path(draft).exists() else None
        if model is None:
            print(f"[server] no checkpoint at {ckpt!r}: model endpoints will return 503")
        pg = cls(model, tok, dmodel, runs_dir, meta={"ckpt": ckpt, "draft": draft, "device": device})
        pg.warmup()
        return pg

    def static_decoders(self, k: int) -> tuple[StaticDecoder, StaticDecoder]:
        """CUDA-graph decoders for target (T=1 and T=k+1) and draft (T=1), captured lazily."""
        if not hasattr(self, "_static"):
            self._static = (StaticDecoder(self.require_model(), (1,)), StaticDecoder(self.draft, (1,)))
        tdec, ddec = self._static
        tdec.ensure(k + 1)
        return tdec, ddec

    def warmup(self) -> None:
        """Pay CUDA context / kernel-selection costs before the first user request."""
        for m in (self.model, self.draft):
            if m is not None:
                list(stream(m, [self.tok.eot_id], 8, temperature=0))

    def require_model(self) -> Transformer:
        if self.model is None:
            raise HTTPException(503, "no model checkpoint loaded - train one with `python -m forge.train`")
        return self.model

    def text(self, tid: int) -> str:
        return self.tok.decode([tid])

    def describe(self, m: Transformer | None) -> dict[str, Any] | None:
        if m is None:
            return None
        meta = getattr(m, "checkpoint_meta", {})
        return {"params": m.num_params(), "config": m.cfg.to_dict(), "step": meta.get("step"),
                "val_loss": meta.get("val_loss"), "tokens_seen": meta.get("tokens_seen"),
                "run_name": meta.get("run_name")}

    @property
    def device(self) -> torch.device:
        m = self.model or self.draft
        return next(m.parameters()).device if m is not None else torch.device("cpu")

    # ------------------------------------------------------------------ analyses

    def cl100k(self):  # noqa: ANN201 - optional dependency
        if self._cl100k is None:
            try:
                import tiktoken

                self._cl100k = tiktoken.get_encoding("cl100k_base")
            except Exception:  # pragma: no cover - offline
                self._cl100k = False
        return self._cl100k or None

    @torch.no_grad()
    def head_scores(self) -> dict[str, Any]:
        """Previous-token, induction, first-token ("sink") and entropy score per head.

        Induction score: on a random token sequence repeated twice, the attention a token in
        the second copy pays to the token *after* its earlier occurrence (Olsson et al. 2022).
        """
        if self._heads is not None:
            return self._heads
        m = self.require_model()
        g = torch.Generator().manual_seed(0)
        L = 48
        rand = torch.randint(300, self.tok.vocab_size - 1, (4, L), generator=g)
        seq = torch.cat([rand, rand], dim=1).to(self.device)
        with self.lock:
            m(seq, capture_attn=True)
            maps = [a.float().cpu() for a in m.attention_maps()]  # each (B, H, 2L, 2L)
        text_ids = self.tok.encode("Once upon a time, there was a little dog named Max. Max liked to run in the "
                                   "park with his friend Lily. One day, Lily found a shiny red ball under a tree.")
        with self.lock:
            m(torch.tensor([text_ids], device=self.device), capture_attn=True)
            tmaps = [a.float().cpu() for a in m.attention_maps()]
        out = []
        idx = torch.arange(L, 2 * L)
        for layer, (a, t) in enumerate(zip(maps, tmaps)):
            T = t.shape[-1]
            rows = torch.arange(1, T)
            prev = t[0, :, rows, rows - 1].mean(-1)
            sink = t[0, :, 1:, 0].mean(-1)
            ent = -(t[0].clamp_min(1e-12).log() * t[0]).sum(-1)[:, 1:].mean(-1)
            induction = a[:, :, idx, idx - L + 1].mean(dim=(0, 2))
            for h in range(a.shape[1]):
                out.append({"layer": layer, "head": h, "prev": float(prev[h]), "induction": float(induction[h]),
                            "sink": float(sink[h]), "entropy": float(ent[h])})
        self._heads = {"heads": out, "n_layers": len(maps), "n_heads": maps[0].shape[1]}
        return self._heads

    @torch.no_grad()
    def embeddings(self, method: str = "pca") -> dict[str, Any]:
        """2-D map of the token embeddings: PCA (instant) or t-SNE on the top-50 PCs.

        t-SNE takes about a minute for 8k points, so it is cached next to the checkpoint.
        """
        if self._emb is None:
            self._emb = {}
        if method in self._emb:
            return self._emb[method]
        m = self.require_model()
        W = m.tok_emb.weight.detach().float().cpu()
        Wc = W - W.mean(0, keepdim=True)
        _, S, V = torch.linalg.svd(Wc, full_matrices=False)
        var = (S[:2] ** 2 / (S**2).sum()).tolist()
        n = W.shape[0]
        base = {"norm": np.round(Wc.norm(dim=1).numpy(), 3).tolist(), "tokens": [self.text(i) for i in range(n)],
                "explained_variance": var, "method": method}
        if method == "tsne":
            cache = Path(str(self.meta.get("ckpt") or "")).with_suffix(".tsne.json") if self.meta.get("ckpt") else None
            if cache is not None and cache.exists():
                xy = np.asarray(json.loads(cache.read_text()))
            else:
                from sklearn.manifold import TSNE

                pcs = (Wc @ V[:50].T).numpy()
                xy = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                          random_state=0).fit_transform(pcs)
                if cache is not None:
                    cache.write_text(json.dumps(np.round(xy, 3).tolist()))
        else:
            xy = (Wc @ V[:2].T).numpy()
        self._emb[method] = {**base, "x": np.round(xy[:, 0], 4).tolist(), "y": np.round(xy[:, 1], 4).tolist()}
        return self._emb[method]


# --------------------------------------------------------------------------- app


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def create_app(pg: Playground) -> FastAPI:
    app = FastAPI(title="Forge playground", docs_url="/api/docs")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {"model": pg.describe(pg.model), "draft": pg.describe(pg.draft), "device": str(pg.device),
                "vocab_size": pg.tok.vocab_size, "torch": torch.__version__,
                "cuda": torch.cuda.get_device_name(0) if pg.device.type == "cuda" else None}

    @app.post("/api/generate")
    def generate(req: GenerateRequest) -> StreamingResponse:
        m = pg.require_model()
        ids = pg.tok.encode(req.prompt) or [pg.tok.eot_id]

        def events() -> Iterator[str]:
            n, t0 = 0, time.perf_counter()
            with pg.lock:
                yield _sse({"type": "prompt", "tokens": [{"id": i, "text": pg.text(i)} for i in ids]})
                for ev in stream(m, ids, req.max_new_tokens, temperature=req.temperature, top_k=req.top_k,
                                 top_p=req.top_p, repetition_penalty=req.repetition_penalty, seed=req.seed,
                                 stop_token=pg.tok.eot_id):
                    n += 1
                    yield _sse({"type": "token", "id": ev.id, "text": pg.text(ev.id), "prob": ev.prob,
                                "entropy": ev.entropy, "t": ev.elapsed, "eot": ev.id == pg.tok.eot_id,
                                "top": [{"id": i, "text": pg.text(i), "prob": p} for i, p in ev.top]})
            dt = time.perf_counter() - t0
            yield _sse({"type": "done", "n": n, "seconds": dt, "tok_per_s": n / max(dt, 1e-9)})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/race")
    def race(req: RaceRequest) -> dict[str, Any]:
        m = pg.require_model()
        if pg.draft is None:
            raise HTTPException(503, "no draft model loaded")
        ids = (pg.tok.encode(req.prompt) or [pg.tok.eot_id])[-200:]
        eot = pg.tok.eot_id

        def fmt_round(t: float, drafted: list[int], n_acc: int, emitted: list[int], kinds: list[str]) -> dict:
            rejected = drafted[n_acc:] if kinds[-1] == "resampled" else []
            return {"t": t, "n_accepted": n_acc, "drafted": len(drafted),
                    "emitted": [{"id": i, "text": pg.text(i), "kind": k} for i, k in zip(emitted, kinds)],
                    "rejected": [{"id": i, "text": pg.text(i)} for i in rejected]}

        with pg.lock:
            if req.engine == "graph":
                if pg.device.type != "cuda":
                    raise HTTPException(400, "the CUDA-graph engine needs a GPU")
                tdec, ddec = pg.static_decoders(req.k)
                fast_generate(m, ids, 4, decoder=tdec)  # warm-up
                fast_speculative(m, pg.draft, ids, 4, k=req.k, decoders=(tdec, ddec))
                trace: list[dict] = []
                spec_out, _ = fast_speculative(m, pg.draft, ids, req.max_new_tokens, k=req.k,
                                               temperature=req.temperature, seed=req.seed or 0,
                                               decoders=(tdec, ddec), trace=trace)
                rounds = [fmt_round(r["t"], r["drafted"], r["n_accepted"], r["emitted"], r["kinds"]) for r in trace]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                base_ids = fast_generate(m, ids, len(spec_out), temperature=req.temperature, seed=req.seed or 0,
                                         decoder=tdec)  # no per-token sync: times are interpolated
                t_total = time.perf_counter() - t0
                base = [{"id": i, "text": pg.text(i), "t": t_total * (j + 1) / len(base_ids)}
                        for j, i in enumerate(base_ids)]
            else:
                # warm-up both paths so neither pays one-off CUDA/allocator costs
                list(stream(m, ids, 4, temperature=0, top_n=0))
                list(speculative_stream(m, pg.draft, ids, 4, k=req.k, temperature=0))
                rounds = [fmt_round(r.elapsed, r.drafted, r.n_accepted, r.emitted, r.kinds)
                          for r in speculative_stream(m, pg.draft, ids, req.max_new_tokens, k=req.k,
                                                      temperature=req.temperature, top_p=req.top_p,
                                                      seed=req.seed, stop_token=eot)]
                n_spec = sum(len(r["emitted"]) for r in rounds)
                base = [{"id": e.id, "text": pg.text(e.id), "t": e.elapsed}
                        for e in stream(m, ids, max(n_spec, 1), temperature=req.temperature, top_p=req.top_p,
                                        seed=req.seed, stop_token=eot, top_n=0)]  # lean path: a fair baseline
        n_spec = sum(len(r["emitted"]) for r in rounds)
        accepted = sum(r["n_accepted"] for r in rounds)
        drafted = sum(r["drafted"] for r in rounds)
        spec_ids = [t["id"] for r in rounds for t in r["emitted"]]
        t_spec = rounds[-1]["t"] if rounds else 0.0
        t_base = base[-1]["t"] if base else 0.0
        return {
            "prompt": [{"id": i, "text": pg.text(i)} for i in ids], "engine": req.engine,
            "speculative": {"rounds": rounds, "tokens": n_spec, "seconds": t_spec,
                            "tok_per_s": n_spec / max(t_spec, 1e-9)},
            "baseline": {"tokens": base, "seconds": t_base, "tok_per_s": len(base) / max(t_base, 1e-9)},
            "stats": {"k": req.k, "rounds": len(rounds), "accepted": accepted,
                      "acceptance_rate": accepted / max(drafted, 1),
                      "tokens_per_round": n_spec / max(len(rounds), 1),
                      "speedup": (t_base / max(len(base), 1)) / max(t_spec / max(n_spec, 1), 1e-9),
                      "identical": [b["id"] for b in base] == spec_ids[:len(base)]},
        }

    @app.post("/api/attention")
    def attention(req: TextRequest) -> dict[str, Any]:
        m = pg.require_model()
        ids = (pg.tok.encode(req.text) or [pg.tok.eot_id])[:64]
        with pg.lock, torch.no_grad():
            m(torch.tensor([ids], device=pg.device), capture_attn=True)
            maps = torch.stack([a[0].float() for a in m.attention_maps()]).cpu()  # (L, H, T, T)
        q = (maps.clamp(0, 1).sqrt() * 255).round().to(torch.uint8).numpy()  # sqrt boosts small weights
        return {"tokens": [pg.text(i) for i in ids], "n_layers": q.shape[0], "n_heads": q.shape[1], "T": len(ids),
                "encoding": "uint8(sqrt(p)*255), row-major (L, H, T, T)",
                "data": base64.b64encode(q.tobytes()).decode()}

    @app.get("/api/heads")
    def heads() -> dict[str, Any]:
        return pg.head_scores()

    @app.post("/api/tokenize")
    def tokenize(req: TextRequest) -> dict[str, Any]:
        ids = pg.tok.encode(req.text)
        n_bytes = len(req.text.encode("utf-8"))
        out: dict[str, Any] = {
            "tokens": [{"id": i, "text": pg.text(i), "bytes": len(pg.tok.token_bytes(i))} for i in ids],
            "n_tokens": len(ids), "n_bytes": n_bytes, "n_chars": len(req.text),
            "bytes_per_token": n_bytes / max(len(ids), 1), "vocab_size": pg.tok.vocab_size,
        }
        enc = pg.cl100k()
        if enc is not None:
            cids = enc.encode(req.text, allowed_special="all")
            out["cl100k"] = {"n_tokens": len(cids), "bytes_per_token": n_bytes / max(len(cids), 1),
                             "tokens": [{"id": i, "text": enc.decode_single_token_bytes(i).decode("utf-8", "replace")}
                                        for i in cids[:2000]]}
        return out

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        out = []
        for d in sorted(pg.runs_dir.glob("*/metrics.jsonl")):
            run = d.parent
            summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
            out.append({"name": run.name, "summary": summary, "active": summary is None,
                        "updated": d.stat().st_mtime})
        return out

    @app.get("/api/runs/{name}")
    def run(name: str, max_points: int = 1500) -> dict[str, Any]:
        run_dir = pg.runs_dir / name
        if not (run_dir / "metrics.jsonl").exists() or "/" in name or "\\" in name or ".." in name:
            raise HTTPException(404, f"unknown run {name!r}")
        train, evals = [], []
        for line in (run_dir / "metrics.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            (train if rec.get("type") == "train" else evals).append(rec)
        if len(train) > max_points:
            stride = math.ceil(len(train) / max_points)
            train = train[::stride] + ([train[-1]] if (len(train) - 1) % stride else [])
        cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else None
        summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else None
        samples = []
        if (run_dir / "samples.jsonl").exists():
            samples = [json.loads(s) for s in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
                       if s.strip()][-1:]
        return {"name": name, "train": train, "eval": evals, "config": cfg, "summary": summary, "samples": samples}

    @app.get("/api/embeddings")
    def embeddings(method: Literal["pca", "tsne"] = "pca") -> JSONResponse:
        return JSONResponse(pg.embeddings(method))

    @app.get("/api/neighbors/{token_id}")
    def neighbors(token_id: int, k: int = 12) -> dict[str, Any]:
        m = pg.require_model()
        W = m.tok_emb.weight.detach().float()
        if not 0 <= token_id < W.shape[0]:
            raise HTTPException(404, "token id out of range")
        Wn = torch.nn.functional.normalize(W, dim=1)
        sims = Wn @ Wn[token_id]
        sims[token_id] = -2.0
        top = torch.topk(sims, k)
        return {"id": token_id, "text": pg.text(token_id),
                "neighbors": [{"id": int(i), "text": pg.text(int(i)), "cos": float(c)}
                              for c, i in zip(top.values.cpu(), top.indices.cpu())]}

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Forge playground server")
    ap.add_argument("--ckpt", default="checkpoints/tinystories-30m/model.safetensors",
                    help="exported weights (python -m forge.export) or a training checkpoint")
    ap.add_argument("--draft", default="checkpoints/draft-6m/model.safetensors")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    pg = Playground.from_paths(args.ckpt, args.tokenizer, args.draft, args.device, Path(args.runs))
    uvicorn.run(create_app(pg), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
