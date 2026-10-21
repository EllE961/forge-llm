"""Static-shape decoding with CUDA graphs.

At 27M parameters a single-token decode step is ~200 tiny kernels; in eager PyTorch the
CPU-side launch overhead dominates and the GPU idles between kernels. This module rewrites
the cached forward with *static shapes* so it can be captured once into a CUDA graph and
replayed with a single launch:

* the KV cache is preallocated (it already is) and written with ``index_copy_`` at a
  position held in a device tensor, not a Python int;
* attention always runs over all ``max_len`` cache slots with a mask
  ``key_pos <= query_pos``, so shapes never change;
* RoPE tables are gathered with ``index_select`` at the device-side positions.

One graph is captured per (model, T): T=1 for plain decoding, T=k+1 for speculative
verification. Token choice (argmax / sampling) stays on the device, so plain greedy
generation needs **no host synchronisation at all** until the very end.

With launch overhead gone, decoding becomes bound by reading the weights from memory -
exactly the regime in which speculative decoding pays off (see ``fast_speculative``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import torch

from forge.generate import sampling_probs
from forge.model import KVCache, Transformer, apply_rope


def static_forward(model: Transformer, tokens: torch.Tensor, pos: torch.Tensor, cache: KVCache) -> torch.Tensor:
    """Forward ``tokens`` (B, T) placed at positions ``pos .. pos+T-1``; returns (B, T, V) logits.

    Shapes depend only on (B, T, cache.max_len), never on ``pos`` - so this is capturable.
    Writes K/V into ``cache`` but does not touch ``cache.length`` (the caller tracks it).
    """
    cfg = model.cfg
    B, T = tokens.shape
    H, KV, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    dev = tokens.device
    positions = pos + torch.arange(T, device=dev)
    cos = model.rope_cos.index_select(0, positions)
    sin = model.rope_sin.index_select(0, positions)
    visible = torch.arange(cache.max_len, device=dev)[None, :] <= positions[:, None]  # (T, L)
    scale = 1.0 / math.sqrt(hd)
    x = model.tok_emb(tokens)
    for li, blk in enumerate(model.layers):
        a = blk.attn
        q, k, v = a.wqkv(blk.attn_norm(x)).split([H * hd, KV * hd, KV * hd], dim=-1)
        q = apply_rope(q.view(B, T, H, hd).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(B, T, KV, hd).transpose(1, 2), cos, sin)
        v = v.view(B, T, KV, hd).transpose(1, 2)
        cache.k[li].index_copy_(2, positions, k.to(cache.k.dtype))
        cache.v[li].index_copy_(2, positions, v.to(cache.v.dtype))
        K, V = cache.k[li].to(q.dtype), cache.v[li].to(q.dtype)
        if KV != H:
            g = H // KV
            K = K[:, :, None].expand(B, KV, g, cache.max_len, hd).reshape(B, H, cache.max_len, hd)
            V = V[:, :, None].expand(B, KV, g, cache.max_len, hd).reshape(B, H, cache.max_len, hd)
        att = (q @ K.transpose(-2, -1)) * scale
        att = att.masked_fill(~visible, float("-inf")).float().softmax(-1).to(V.dtype)
        y = (att @ V).transpose(1, 2).reshape(B, T, H * hd)
        x = x + a.wo(y)
        x = x + blk.ffn(blk.ffn_norm(x))
    return model.lm_head(model.norm(x))


class StaticDecoder:
    """A model + its KV cache + one captured CUDA graph per input length T.

    ``use_graphs=False`` runs the same static code eagerly (used on CPU and in tests).
    """

    def __init__(self, model: Transformer, lengths: Sequence[int] = (1,), use_graphs: bool | None = None):
        self.model = model
        self.device = next(model.parameters()).device
        self.use_graphs = self.device.type == "cuda" if use_graphs is None else use_graphs
        self.cache = model.new_cache(1)
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if self.use_graphs:
            for T in lengths:
                self._capture(T)
            self.cache.k.zero_()
            self.cache.v.zero_()

    def ensure(self, T: int) -> None:
        """Capture a graph for input length T if missing (call before ``prefill``: capture
        warm-up writes scratch values into the first T cache slots)."""
        if self.use_graphs and T not in self.graphs:
            self._capture(T)

    @torch.no_grad()
    def _capture(self, T: int) -> None:
        tokens = torch.zeros(1, T, dtype=torch.long, device=self.device)
        pos = torch.zeros((), dtype=torch.long, device=self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up allocations/kernels off the capture stream
            for _ in range(3):
                static_forward(self.model, tokens, pos, self.cache)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):  # private memory pool per graph: no aliasing between graphs
            out = static_forward(self.model, tokens, pos, self.cache)
        self.graphs[T] = (graph, tokens, pos, out)

    @torch.no_grad()
    def prefill(self, ids: Sequence[int]) -> torch.Tensor:
        """Run the prompt eagerly into the cache; returns the logits of the last position."""
        t = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        self.cache.length = 0
        return self.model(t, cache=self.cache, last_only=True)[0][0, -1]

    @torch.no_grad()
    def __call__(self, tokens: torch.Tensor, pos: int) -> torch.Tensor:
        """Logits (T, V) for ``tokens`` (T,) at positions pos..pos+T-1 (device tensors in, out)."""
        T = tokens.numel()
        if self.use_graphs and T in self.graphs:
            graph, tok_buf, pos_buf, out = self.graphs[T]
            tok_buf.copy_(tokens.view(1, T))
            pos_buf.fill_(pos)
            graph.replay()
            return out[0]
        p = torch.tensor(pos, dtype=torch.long, device=self.device)
        return static_forward(self.model, tokens.view(1, T), p, self.cache)[0]


def _choose(logits: torch.Tensor, temperature: float, gen: torch.Generator | None) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    return torch.multinomial(sampling_probs(logits, temperature), 1, generator=gen)


@torch.no_grad()
def fast_generate(model: Transformer, prompt: Sequence[int], max_new_tokens: int, temperature: float = 0.0,
                  seed: int = 0, decoder: StaticDecoder | None = None) -> list[int]:
    """Plain decoding through the static/graphed path. No host sync until the end."""
    dec = decoder or StaticDecoder(model, (1,))
    gen = torch.Generator(device=dec.device)
    gen.manual_seed(seed)
    if len(prompt) + max_new_tokens > model.cfg.max_seq_len:
        raise ValueError("prompt + new tokens exceed the context window")
    tok = _choose(dec.prefill(prompt), temperature, gen)
    out = [tok]
    pos = len(prompt)
    for _ in range(max_new_tokens - 1):
        tok = _choose(dec(tok, pos)[-1], temperature, gen)
        out.append(tok)
        pos += 1
    return torch.cat(out).tolist()


@torch.no_grad()
def fast_speculative(target: Transformer, draft: Transformer, prompt: Sequence[int], max_new_tokens: int,
                     k: int = 4, temperature: float = 0.0, seed: int = 0,
                     decoders: tuple[StaticDecoder, StaticDecoder] | None = None,
                     trace: list[dict] | None = None) -> tuple[list[int], dict]:
    """Speculative decoding on graphed decoders (same acceptance rule as ``forge.speculative``).

    Target cache invariant: it holds every emitted token except the newest one, so each
    verify call is exactly ``[newest, d_1..d_k]`` - a fixed length, hence one graph.
    If ``trace`` is a list, one record per round (drafted / accepted / time) is appended -
    collected on the device and converted once at the end, so tracing adds no syncs.
    """
    from forge.speculative import accept_reject_device

    t_dec, d_dec = decoders or (StaticDecoder(target, (k + 1,)), StaticDecoder(draft, (1,)))
    dev = t_dec.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    if len(prompt) + max_new_tokens + k + 1 > min(target.cfg.max_seq_len, draft.cfg.max_seq_len):
        raise ValueError("prompt + new tokens exceed the context window")
    t0 = time.perf_counter()
    # prefill both; `newest` is the latest emitted token, not yet in the target cache
    newest = _choose(t_dec.prefill(prompt), temperature, gen)
    d_dec.prefill(prompt)
    n = len(prompt)  # position of `newest`
    pending_draft: torch.Tensor | None = None  # d_k when a whole round was accepted
    out: list[torch.Tensor] = [newest]
    produced, rounds, accepted = 1, 0, 0
    records: list[tuple[torch.Tensor, int, torch.Tensor, float]] = []
    while produced < max_new_tokens:
        # draft: if the last round accepted all k, the draft has not seen d_k yet
        if pending_draft is not None:
            d_dec(pending_draft, n - 1)
        cur, drafts, q_rows = newest, [], []
        for j in range(k):
            q = sampling_probs(d_dec(cur, n + j)[-1], temperature)
            cur = q.argmax().view(1) if temperature <= 0 else torch.multinomial(q, 1, generator=gen)
            drafts.append(cur)
            q_rows.append(q)
        drafted = torch.cat(drafts)
        # verify [newest, d_1..d_k] in one target pass; accept/reject without leaving the GPU
        p = sampling_probs(t_dec(torch.cat([newest, drafted]), n), temperature)
        n_acc_t, nxt = accept_reject_device(p, torch.stack(q_rows), drafted, gen)
        n_acc = int(n_acc_t)  # the round's single host synchronisation
        if trace is not None:
            records.append((drafted, n_acc, nxt, time.perf_counter() - t0))
        out.extend([drafted[:n_acc], nxt])
        pending_draft = drafted[k - 1:] if n_acc == k else None
        newest = nxt
        n += n_acc + 1
        produced += n_acc + 1
        rounds += 1
        accepted += n_acc
    tokens = torch.cat(out).tolist()[:max_new_tokens]
    dt = time.perf_counter() - t0
    if trace is not None:
        first_t = records[0][3] if records else dt  # the prefill token surfaces with the first sync
        trace.append({"t": first_t, "drafted": [], "n_accepted": 0, "emitted": tokens[:1], "kinds": ["target"]})
        for drafted, n_acc, nxt, t in records:
            d = drafted.tolist()
            trace.append({"t": t, "drafted": d, "n_accepted": n_acc, "emitted": d[:n_acc] + nxt.tolist(),
                          "kinds": ["accepted"] * n_acc + ["bonus" if n_acc == k else "resampled"]})
    return tokens, {"rounds": rounds, "acceptance_rate": accepted / max(rounds * k, 1),
                    "tokens_per_round": (produced - 1) / max(rounds, 1), "seconds": dt}  # excl. prefill token
