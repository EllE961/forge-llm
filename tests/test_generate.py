from __future__ import annotations

import torch

from forge.generate import apply_repetition_penalty, filter_logits, generate, sampling_probs, stream

from .conftest import tiny_model


def test_top_k_keeps_exactly_k():
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
    kept = torch.isfinite(filter_logits(logits, top_k=2))
    assert kept.tolist() == [False, True, False, False, True]


def test_top_p_keeps_smallest_nucleus():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log()
    assert torch.isfinite(filter_logits(logits, top_p=0.75)).tolist() == [True, True, False, False]
    assert torch.isfinite(filter_logits(logits, top_p=0.8)).tolist() == [True, True, False, False]
    assert torch.isfinite(filter_logits(logits, top_p=0.81)).tolist() == [True, True, True, False]
    assert torch.isfinite(filter_logits(logits, top_p=0.01)).tolist() == [True, False, False, False]


def test_sampling_probs_greedy_is_one_hot():
    p = sampling_probs(torch.tensor([0.1, 2.0, -1.0]), temperature=0)
    assert p.tolist() == [0.0, 1.0, 0.0]


def test_repetition_penalty_lowers_seen_tokens():
    logits = torch.tensor([2.0, -1.0, 0.5])
    out = apply_repetition_penalty(logits, torch.tensor([0, 1]), 2.0)
    assert out.tolist() == [1.0, -2.0, 0.5]


def test_greedy_with_cache_equals_without_cache():
    m = tiny_model()
    prompt = [5, 17, 42, 99]
    a = generate(m, prompt, 30, temperature=0)
    b = generate(m, prompt, 30, temperature=0, use_cache=False)
    assert a == b and len(a) == 30


def test_sampling_is_seeded_and_events_are_consistent():
    m = tiny_model()
    ev1 = list(stream(m, [1, 2, 3], 20, temperature=1.0, seed=7))
    ev2 = list(stream(m, [1, 2, 3], 20, temperature=1.0, seed=7))
    assert [e.id for e in ev1] == [e.id for e in ev2]
    for e in ev1:
        assert 0 < e.prob <= 1 and e.entropy >= 0 and len(e.top) == 5
        assert e.top[0][1] >= e.prob - 1e-6 or e.id == e.top[0][0]


def test_stop_token_and_context_overflow():
    m = tiny_model()
    out = list(stream(m, [1, 2, 3], 300, temperature=1.0, seed=0))  # > max_seq_len: sliding window
    assert len(out) == 300
    first = out[0].id
    stopped = generate(m, [1, 2, 3], 50, temperature=1.0, seed=0, stop_token=first)
    assert stopped == []
