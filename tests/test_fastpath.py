from __future__ import annotations

import pytest
import torch

from forge.fastpath import StaticDecoder, fast_generate, fast_speculative
from forge.generate import generate

from .conftest import tiny_model


def test_static_forward_matches_cached_forward():
    m = tiny_model()
    ids = torch.randint(0, 320, (1, 30))
    full = m(ids)[0][0]
    dec = StaticDecoder(m, use_graphs=False)
    last = dec.prefill(ids[0, :10].tolist())
    torch.testing.assert_close(last, full[9], atol=1e-5, rtol=1e-5)
    rows = [dec(ids[0, t:t + 1], t)[-1] for t in range(10, 20)]
    rows.append(dec(ids[0, 20:30], 20)[-1])  # multi-token (verify-style) call
    torch.testing.assert_close(torch.stack(rows[:-1]), full[10:20], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rows[-1], full[29], atol=1e-5, rtol=1e-5)


def test_fast_generate_matches_reference_greedy():
    m = tiny_model()
    prompt = [5, 6, 7, 8]
    assert fast_generate(m, prompt, 40, decoder=StaticDecoder(m, use_graphs=False)) == generate(m, prompt, 40,
                                                                                              temperature=0)


def test_fast_speculative_greedy_is_exact():
    target, draft = tiny_model(seed=0), tiny_model(seed=2, n_layers=1)
    prompt = [9, 10, 11]
    ref = generate(target, prompt, 50, temperature=0)
    for k in (1, 3, 5):
        decs = (StaticDecoder(target, use_graphs=False), StaticDecoder(draft, use_graphs=False))
        out, stats = fast_speculative(target, draft, prompt, 50, k=k, decoders=decs)
        assert out == ref, k
        assert 0 <= stats["acceptance_rate"] <= 1


def test_fast_speculative_identical_draft_accepts_all_when_sampling():
    m = tiny_model()
    decs = (StaticDecoder(m, use_graphs=False), StaticDecoder(m, use_graphs=False))
    out, stats = fast_speculative(m, m, [1, 2, 3], 41, k=4, temperature=1.0, seed=3, decoders=decs)
    assert len(out) == 41 and stats["acceptance_rate"] == 1.0 and stats["tokens_per_round"] == 5.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_cuda_graph_replay_matches_eager():
    m = tiny_model().cuda()
    prompt = [1, 2, 3, 4, 5]
    eager = fast_generate(m, prompt, 30, decoder=StaticDecoder(m, use_graphs=False))
    graphed = fast_generate(m, prompt, 30, decoder=StaticDecoder(m, (1,), use_graphs=True))
    assert eager == graphed
