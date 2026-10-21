"""Autoregressive sampling with a KV cache.

``stream`` is the core: it prefills the prompt once, then feeds one token per step and
yields a :class:`TokenEvent` per generated token carrying everything the playground
visualises (the model's own probability for the token, the entropy of the next-token
distribution and the top-n alternatives).

Sampling pipeline applied to the logits each step::

    repetition penalty (CTRL, Keskar et al. 2019) -> temperature -> top-k -> top-p (nucleus,
    Holtzman et al. 2019) -> softmax -> multinomial      (temperature 0 = greedy)
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch

from forge.model import Transformer


@dataclass
class TokenEvent:
    id: int
    prob: float  # model probability at temperature 1, before any filtering
    entropy: float  # nats, of the unfiltered next-token distribution
    top: list[tuple[int, float]] = field(default_factory=list)  # top-n (id, prob) alternatives
    elapsed: float = 0.0  # seconds since generation started


def apply_repetition_penalty(logits: torch.Tensor, prev: torch.Tensor, penalty: float) -> torch.Tensor:
    """Divide positive / multiply negative logits of already-seen tokens by ``penalty``."""
    if penalty == 1.0 or prev.numel() == 0:
        return logits
    logits = logits.clone()
    seen = torch.unique(prev)
    vals = logits[..., seen]
    logits[..., seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Mask (to -inf) everything outside the top-k set and outside the smallest nucleus with mass >= top_p."""
    logits = logits.clone()
    if top_k is not None and 0 < top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits[logits < kth] = float("-inf")
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        # drop a token if the mass *before* it already reaches top_p (always keeps the argmax)
        drop_sorted = (cum - sorted_logits.softmax(-1)) >= top_p
        drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
        logits[drop] = float("-inf")
    return logits


def sampling_probs(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                   top_p: float | None = None) -> torch.Tensor:
    """The distribution actually sampled from. Temperature 0 gives a one-hot argmax."""
    logits = logits.float()
    if temperature <= 0:
        out = torch.zeros_like(logits)
        out.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return out
    return filter_logits(logits / temperature, top_k, top_p).softmax(-1)


def _event(logits: torch.Tensor, tid: int, top_n: int, t0: float) -> TokenEvent:
    if top_n <= 0:  # lean path for benchmarks / plain generation: no extra kernels or syncs
        return TokenEvent(id=tid, prob=float("nan"), entropy=float("nan"), elapsed=time.perf_counter() - t0)
    probs = logits.float().softmax(-1)
    ent = -(probs * probs.clamp_min(1e-12).log()).sum().item()
    tp, ti = torch.topk(probs, top_n)
    return TokenEvent(id=tid, prob=probs[tid].item(), entropy=ent,
                      top=list(zip(ti.tolist(), tp.tolist())), elapsed=time.perf_counter() - t0)


@torch.no_grad()
def stream(
    model: Transformer,
    prompt: Sequence[int],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    stop_token: int | None = None,
    top_n: int = 5,
    use_cache: bool = True,
) -> Iterator[TokenEvent]:
    """Yield generated tokens one at a time."""
    device = next(model.parameters()).device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed if seed is not None else int(time.time_ns() % (2**31)))
    max_len = model.cfg.max_seq_len
    ids = list(prompt)[-(max_len - 1):] or [stop_token or 0]
    seq = torch.tensor(ids, dtype=torch.long, device=device)
    cache = model.new_cache(1) if use_cache else None
    offset = 0  # index in `seq` of the token stored at cache position 0
    t0 = time.perf_counter()

    def forward_new(tokens: torch.Tensor) -> torch.Tensor:
        nonlocal cache, offset
        if cache is None:
            return model(tokens[-max_len:][None], last_only=True)[0][0, -1]
        n_new = len(tokens) - offset - cache.length
        if cache.length + n_new > max_len:  # window full: keep the most recent half, re-prefill
            cache = model.new_cache(1)
            offset = len(tokens) - max_len // 2
            return model(tokens[offset:][None], cache=cache, last_only=True)[0][0, -1]
        return model(tokens[offset + cache.length:][None], cache=cache, last_only=True)[0][0, -1]

    logits = forward_new(seq)
    for _ in range(max_new_tokens):
        adj = apply_repetition_penalty(logits, seq, repetition_penalty)
        if temperature <= 0:
            tid = int(adj.argmax())
        else:
            tid = int(torch.multinomial(sampling_probs(adj, temperature, top_k, top_p), 1, generator=gen).item())
        yield _event(logits, tid, top_n, t0)
        if stop_token is not None and tid == stop_token:
            return
        seq = torch.cat([seq, seq.new_tensor([tid])])
        logits = forward_new(seq)


def generate(model: Transformer, prompt: Sequence[int], max_new_tokens: int = 200, **kw) -> list[int]:
    """Convenience wrapper returning only the generated ids (stop token excluded)."""
    kw.setdefault("top_n", 0)
    stop = kw.get("stop_token")
    out = [e.id for e in stream(model, prompt, max_new_tokens, **kw)]
    return out[:-1] if out and stop is not None and out[-1] == stop else out
