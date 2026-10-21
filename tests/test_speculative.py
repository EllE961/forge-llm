from __future__ import annotations

from collections import Counter

import pytest
import torch

from forge.generate import generate, sampling_probs
from forge.speculative import accept_reject, accept_reject_device, speculative_generate, speculative_stream

from .conftest import tiny_model


def _tv(counts: Counter, probs: dict, n: int) -> float:
    keys = set(counts) | set(probs)
    return 0.5 * sum(abs(counts.get(x, 0) / n - probs.get(x, 0.0)) for x in keys)


def _host_rule(p, q, drafted, gen):
    n_acc, nxt, _ = accept_reject(p, q, drafted, gen)
    return n_acc, nxt


def _device_rule(p, q, drafted, gen):
    n_acc, nxt = accept_reject_device(p, q, torch.tensor(drafted), gen)
    return int(n_acc), int(nxt)


@pytest.mark.parametrize("rule", [_host_rule, _device_rule], ids=["host", "device"])
def test_accept_reject_first_token_marginal_is_target(rule):
    """Synthetic distributions: P(first emitted token) must equal p_0 for any draft q."""
    torch.manual_seed(0)
    V, k, n = 6, 3, 20000
    p = torch.softmax(torch.randn(k + 1, V) * 2, -1)
    q = torch.softmax(torch.randn(k, V) * 2, -1)
    gen = torch.Generator().manual_seed(1)
    counts: Counter = Counter()
    for _ in range(n):
        drafted = [int(torch.multinomial(q[i], 1, generator=gen)) for i in range(k)]
        n_acc, nxt = rule(p, q, drafted, gen)
        counts[drafted[0] if n_acc > 0 else nxt] += 1
    target = {i: float(p[0, i]) for i in range(V)}
    naive = {i: float(q[0, i]) for i in range(V)}
    assert _tv(counts, target, n) < 0.02
    assert _tv(counts, naive, n) > 0.1  # the test would catch "just use the draft"


def test_greedy_speculative_equals_greedy_target():
    target, draft = tiny_model(seed=0), tiny_model(seed=1, n_layers=1)
    prompt = [3, 14, 15, 92, 65]
    for k in (1, 2, 4, 7):
        spec, _ = speculative_generate(target, draft, prompt, 40, k=k, temperature=0)
        assert spec == generate(target, prompt, 40, temperature=0), k


def test_identical_draft_accepts_everything():
    m = tiny_model()
    _, stats = speculative_generate(m, m, [1, 2, 3], 40, k=4, temperature=1.0, seed=0)
    assert stats["acceptance_rate"] == 1.0
    assert stats["tokens_per_round"] == 5.0


def test_sampled_joint_distribution_matches_target_model():
    """End to end: the joint law of the first two sampled tokens equals the target's."""
    V = 8
    target, draft = tiny_model(vocab_size=V, seed=0), tiny_model(vocab_size=V, seed=5, n_layers=1)
    prompt = [1, 2, 3]
    with torch.no_grad():
        p1 = sampling_probs(target(torch.tensor([prompt]))[0][0, -1], 1.0)
        exact = {}
        for a in range(V):
            p2 = sampling_probs(target(torch.tensor([prompt + [a]]))[0][0, -1], 1.0)
            for b in range(V):
                exact[(a, b)] = float(p1[a] * p2[b])
    n = 3000
    counts: Counter = Counter()
    for s in range(n):
        out: list[int] = []
        for r in speculative_stream(target, draft, prompt, 2, k=3, temperature=1.0, seed=s):
            out.extend(r.emitted)
        counts[tuple(out[:2])] += 1
    assert _tv(counts, exact, n) < 0.08


def test_rounds_report_consistent_stats():
    target, draft = tiny_model(seed=0), tiny_model(seed=1, n_layers=1)
    rounds = list(speculative_stream(target, draft, [7, 8, 9], 50, k=4, temperature=0.8, seed=3))
    assert sum(len(r.emitted) for r in rounds) == 50
    for r in rounds:
        assert len(r.emitted) == r.n_accepted + 1 or r is rounds[-1]
        assert r.kinds[: r.n_accepted] == ["accepted"] * r.n_accepted
        assert r.emitted[: r.n_accepted] == r.drafted[: r.n_accepted]
