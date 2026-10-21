"""Speculative decoding (Leviathan et al. 2023; Chen et al. 2023).

A small *draft* model proposes ``k`` tokens autoregressively; the large *target* model
scores all of them in a single forward pass. Draft token ``x_i`` (sampled from the
draft distribution ``q_i``) is accepted with probability ``min(1, p_i(x_i) / q_i(x_i))``.
At the first rejection we sample a replacement from the residual distribution

    r(x) = max(0, p_i(x) - q_i(x)) / sum_y max(0, p_i(y) - q_i(y))

and stop; if all ``k`` are accepted we get a free "bonus" token from ``p_{k+1}``. For
every position the emitted token is distributed *exactly* as ``p_i`` - speculative
decoding changes speed, never the output distribution:

    P(emit x) = q(x) min(1, p(x)/q(x)) + (1 - sum_y min(p(y), q(y))) r(x)
              = min(p(x), q(x)) + max(0, p(x) - q(x)) = p(x).

Greedy decoding falls out of the same code: with temperature 0 both distributions are
one-hot, so a draft token is accepted iff it equals the target argmax and the residual
is the target argmax - outputs are identical to plain greedy decoding.

Both models keep a KV cache; "rollback" after a rejection is just resetting the cache
length to the number of positions whose tokens survived.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch

from forge.generate import sampling_probs
from forge.model import Transformer


@dataclass
class SpecRound:
    """One draft-then-verify round."""

    drafted: list[int]
    n_accepted: int
    emitted: list[int]
    kinds: list[str]  # per emitted token: "accepted" | "resampled" | "bonus"
    target_probs: list[float] = field(default_factory=list)  # p(emitted token) under the target
    elapsed: float = 0.0


def accept_reject(p: torch.Tensor, q: torch.Tensor, drafted: Sequence[int] | torch.Tensor,
                  generator: torch.Generator | None = None) -> tuple[int, int, str]:
    """Run the acceptance test for one round.

    All k accept/reject coins are drawn and compared in one vectorised step, so a round
    costs a single device->host sync instead of one per drafted token (uniforms past the
    first rejection are simply unused, which leaves the distribution unchanged).

    Args:
        p: (k+1, V) target sampling distributions for the k drafted positions + 1 bonus.
        q: (k, V) draft sampling distributions the drafted tokens were sampled from.
        drafted: the k drafted token ids.

    Returns:
        ``(n_accepted, next_token, kind)`` where ``next_token`` is the resampled token
        (``kind="resampled"``) or the bonus token (``kind="bonus"``).
    """
    dev = p.device
    xs = torch.as_tensor(drafted, dtype=torch.long, device=dev).view(-1)
    k = xs.numel()
    if k:
        rows = torch.arange(k, device=dev)
        px, qx = p[rows, xs], q[rows, xs]
        u = torch.rand(k, generator=generator, device=dev)
        ok = ((qx > 0) & (u * qx <= px)).tolist()  # u <= p/q without dividing
        n_acc = ok.index(False) if False in ok else k
    else:
        n_acc = 0
    if n_acc < k:
        residual = (p[n_acc] - q[n_acc]).clamp_min(0.0)
        total = residual.sum()
        dist = torch.where(total > 0, residual / total.clamp_min(1e-30), p[n_acc])
        return n_acc, int(torch.multinomial(dist, 1, generator=generator).item()), "resampled"
    return k, int(torch.multinomial(p[k], 1, generator=generator).item()), "bonus"


def accept_reject_device(p: torch.Tensor, q: torch.Tensor, drafted: torch.Tensor,
                         generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Same rule as :func:`accept_reject`, but branch-free and entirely on the device.

    Padding ``q`` with an all-zero row makes the bonus case a special case of the residual
    formula (``max(0, p_k - 0) = p_k``), so the whole round needs no host round-trip.
    Returns ``(n_accepted, next_token)`` as device tensors (0-d and shape (1,)).
    """
    k = drafted.numel()
    dev = p.device
    rows = torch.arange(k, device=dev)
    px, qx = p[rows, drafted], q[rows, drafted]
    u = torch.rand(k, generator=generator, device=dev)
    ok = (qx > 0) & (u * qx <= px)
    first_fail = (~ok).to(torch.int64).argmax()  # 0 when everything was accepted
    n_acc = first_fail + ok.all().to(torch.int64) * (k - first_fail)
    q_pad = torch.cat([q, torch.zeros_like(p[:1])])
    residual = (p[n_acc] - q_pad[n_acc]).clamp_min(0.0)
    total = residual.sum()
    dist = torch.where(total > 0, residual / total.clamp_min(1e-30), p[n_acc])
    return n_acc, torch.multinomial(dist, 1, generator=generator)


@torch.no_grad()
def speculative_stream(
    target: Transformer,
    draft: Transformer,
    prompt: Sequence[int],
    max_new_tokens: int = 200,
    k: int = 4,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    stop_token: int | None = None,
) -> Iterator[SpecRound]:
    """Yield one :class:`SpecRound` per draft/verify iteration."""
    if target.cfg.vocab_size != draft.cfg.vocab_size:
        raise ValueError("target and draft must share a vocabulary")
    device = next(target.parameters()).device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed if seed is not None else int(time.time_ns() % (2**31)))
    max_len = min(target.cfg.max_seq_len, draft.cfg.max_seq_len)
    seq = list(prompt)[-(max_len // 2):]
    t_cache, d_cache = target.new_cache(1), draft.new_cache(1)
    produced = 0
    t0 = time.perf_counter()

    def as_tensor(ids: list[int]) -> torch.Tensor:
        return torch.tensor(ids, dtype=torch.long, device=device)[None]

    while produced < max_new_tokens:
        k_eff = min(k, max_new_tokens - produced - 1, max_len - len(seq) - 1)
        if k_eff < 0:
            break
        # 1) draft k tokens (each cache always holds seq[:cache.length]). Drafted ids stay on
        #    the device and are fed straight back in: no host sync inside the draft loop.
        draft_ids: list[torch.Tensor] = []
        q_rows: list[torch.Tensor] = []
        inp = as_tensor(seq[d_cache.length:])
        for _ in range(k_eff):
            logits = draft(inp, cache=d_cache, last_only=True)[0][0, -1]
            q = sampling_probs(logits, temperature, top_k, top_p)
            x = torch.multinomial(q, 1, generator=gen) if temperature > 0 else q.argmax().view(1)
            draft_ids.append(x)
            q_rows.append(q)
            inp = x.view(1, 1)
        drafted_t = torch.cat(draft_ids) if draft_ids else torch.empty(0, dtype=torch.long, device=device)
        # 2) verify all drafted tokens with one target forward
        n_ctx = len(seq) - t_cache.length
        logits = target(torch.cat([as_tensor(seq[t_cache.length:])[0], drafted_t])[None], cache=t_cache)[0][0]
        p = sampling_probs(logits[n_ctx - 1:], temperature, top_k, top_p)  # (k_eff + 1, V)
        q_mat = torch.stack(q_rows) if q_rows else p[:0]
        n_acc, nxt, kind = accept_reject(p, q_mat, drafted_t, gen)
        drafted = drafted_t.tolist()
        emitted = drafted[:n_acc] + [nxt]
        kinds = ["accepted"] * n_acc + [kind]
        tprobs = p[torch.arange(len(emitted), device=device), torch.tensor(emitted, device=device)].tolist()
        # 3) roll both caches back to the surviving prefix
        valid = len(seq) + n_acc
        t_cache.length = min(t_cache.length, valid)
        d_cache.length = min(d_cache.length, valid)
        if stop_token is not None and stop_token in emitted:
            cut = emitted.index(stop_token) + 1
            emitted, kinds, tprobs = emitted[:cut], kinds[:cut], tprobs[:cut]
        seq.extend(emitted)
        produced += len(emitted)
        yield SpecRound(drafted=drafted, n_accepted=n_acc, emitted=emitted, kinds=kinds,
                        target_probs=tprobs, elapsed=time.perf_counter() - t0)
        if stop_token is not None and emitted and emitted[-1] == stop_token:
            return


def speculative_generate(target: Transformer, draft: Transformer, prompt: Sequence[int],
                         max_new_tokens: int = 200, **kw) -> tuple[list[int], dict[str, float]]:
    """Generate and return ``(token_ids, stats)``."""
    out: list[int] = []
    drafted = accepted = rounds = 0
    t0 = time.perf_counter()
    for r in speculative_stream(target, draft, prompt, max_new_tokens, **kw):
        out.extend(r.emitted)
        drafted += len(r.drafted)
        accepted += r.n_accepted
        rounds += 1
    dt = time.perf_counter() - t0
    stats = {"tokens": len(out), "rounds": rounds, "drafted": drafted, "accepted": accepted,
             "acceptance_rate": accepted / max(drafted, 1), "tokens_per_round": len(out) / max(rounds, 1),
             "seconds": dt, "tok_per_s": len(out) / max(dt, 1e-9)}
    return out, stats
