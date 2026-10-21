from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

import forge.model as fm
from forge.attention import chunked_causal_attention
from forge.model import ModelConfig, Transformer, apply_rope, rope_tables

from .conftest import tiny_model


def test_kv_cache_matches_full_forward():
    m = tiny_model()
    x = torch.randint(0, 320, (2, 40))
    full, _ = m(x)
    cache = m.new_cache(2)
    chunks = [m(x[:, :17], cache=cache)[0]]  # prefill, then mixed-size steps
    for a, b in [(17, 18), (18, 25), (25, 26), (26, 40)]:
        chunks.append(m(x[:, a:b], cache=cache)[0])
    assert cache.length == 40
    torch.testing.assert_close(torch.cat(chunks, 1), full, atol=1e-5, rtol=1e-5)


def test_cache_rollback_overwrites_stale_entries():
    m = tiny_model()
    x = torch.randint(0, 320, (1, 30))
    y = x.clone()
    y[0, 20:] = (y[0, 20:] + 7) % 320
    cache = m.new_cache(1)
    m(x, cache=cache)
    cache.length = 20  # roll back and continue with a different suffix
    out_b = m(y[:, 20:], cache=cache)[0]
    torch.testing.assert_close(out_b, m(y)[0][:, 20:], atol=1e-5, rtol=1e-5)


def test_causality():
    m = tiny_model()
    x = torch.randint(0, 320, (1, 32))
    y = x.clone()
    y[0, 20:] = torch.randint(0, 320, (12,))
    torch.testing.assert_close(m(x)[0][:, :20], m(y)[0][:, :20])


def test_rope_scores_depend_only_on_relative_position():
    hd = 16
    cos, sin = rope_tables(hd, 64)
    q, k = torch.randn(1, 1, 1, hd), torch.randn(1, 1, 1, hd)

    def score(m: int, n: int) -> float:
        qm = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
        kn = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        return float((qm * kn).sum())

    for m, n, s in [(5, 2, 10), (30, 30, 7), (3, 9, 40)]:
        assert math.isclose(score(m, n), score(m + s, n + s), rel_tol=1e-4, abs_tol=1e-5)
    assert not math.isclose(score(5, 2), score(5, 3), rel_tol=1e-3)


def test_gqa_equals_mha_with_repeated_kv_heads():
    gqa = tiny_model(n_heads=4, n_kv_heads=2)
    mha = tiny_model(n_heads=4, n_kv_heads=4, seed=1)
    sd = gqa.state_dict()
    hd, g = gqa.cfg.head_dim, 2
    for i in range(gqa.cfg.n_layers):
        w = sd[f"layers.{i}.attn.wqkv.weight"]
        q, k, v = w.split([4 * hd, 2 * hd, 2 * hd])
        rep = lambda t: t.view(2, hd, -1).repeat_interleave(g, dim=0).reshape(4 * hd, -1)
        sd[f"layers.{i}.attn.wqkv.weight"] = torch.cat([q, rep(k), rep(v)])
    mha.load_state_dict(sd)
    x = torch.randint(0, 320, (2, 24))
    torch.testing.assert_close(gqa(x)[0], mha(x)[0], atol=1e-5, rtol=1e-5)


def test_chunked_attention_matches_reference_forward_and_backward():
    torch.manual_seed(0)
    B, H, T, D = 2, 3, 45, 8
    q, k, v = (torch.randn(B, H, T, D, dtype=torch.float64, requires_grad=True) for _ in range(3))
    s = (q @ k.transpose(-2, -1)) / math.sqrt(D)
    ref = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf")).softmax(-1) @ v
    out = chunked_causal_attention(q, k, v, 1 / math.sqrt(D), block=16)
    torch.testing.assert_close(out, ref)
    g = torch.randn_like(out)
    for a, b in zip(torch.autograd.grad(out, (q, k, v), g), torch.autograd.grad(ref, (q, k, v), g)):
        torch.testing.assert_close(a, b)


def test_model_training_path_matches_naive_attention(monkeypatch):
    m = tiny_model().train()
    x, y = torch.randint(0, 320, (2, 48)), torch.randint(0, 320, (2, 48))
    grads = []
    for flag in (True, False):
        monkeypatch.setattr(fm, "USE_CHUNKED_ATTENTION", flag)
        m.zero_grad()
        loss = m(x, y)[1]
        loss.backward()
        grads.append((loss.item(), m.layers[0].attn.wqkv.weight.grad.clone()))
    assert math.isclose(grads[0][0], grads[1][0], rel_tol=1e-6)
    torch.testing.assert_close(grads[0][1], grads[1][1], atol=1e-6, rtol=1e-4)


def test_attention_capture_rows_are_distributions():
    m = tiny_model()
    m(torch.randint(0, 320, (1, 12)), capture_attn=True)
    maps = m.attention_maps()
    assert len(maps) == m.cfg.n_layers and maps[0].shape == (1, 4, 12, 12)
    torch.testing.assert_close(maps[0].sum(-1), torch.ones(1, 4, 12))
    assert torch.all(maps[0].triu(1) == 0)  # causal


def test_overfits_single_batch():
    torch.manual_seed(0)
    m = tiny_model(vocab_size=64).train()
    x = torch.randint(0, 64, (4, 32))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        loss = m(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_param_accounting():
    m = Transformer(ModelConfig())
    assert 25e6 < m.num_params() < 30e6  # the 27M config used for TinyStories
    assert m.num_params() - m.num_params(non_embedding=True) == 8192 * 512
    assert m.flops_per_token() > 6 * m.num_params(non_embedding=True)


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(d_model=64, n_heads=4, n_kv_heads=3)


def test_trainer_checkpoint_and_resume(tmp_path):
    from forge.train import TrainConfig, Trainer

    data = np.random.default_rng(0).integers(0, 64, size=5000).astype(np.uint16)
    data.tofile(tmp_path / "train.bin")
    data[:2000].tofile(tmp_path / "val.bin")
    mcfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=32)
    tcfg = TrainConfig(run_name="t", ckpt_dir=str(tmp_path / "ck"), runs_dir=str(tmp_path / "runs"),
                       train_bin=str(tmp_path / "train.bin"), val_bin=str(tmp_path / "val.bin"),
                       tokenizer=str(tmp_path / "missing.json"), batch_size=4, seq_len=32, max_steps=6,
                       warmup_steps=2, eval_interval=3, eval_batches=2, log_interval=1, ckpt_interval=3,
                       device="cpu")
    summary = Trainer(mcfg, tcfg).train()
    assert summary["steps"] == 6
    assert (tmp_path / "ck" / "t" / "best.pt").exists()
    recs = [json.loads(line) for line in (tmp_path / "runs" / "t" / "metrics.jsonl").read_text().splitlines()]
    assert {r["type"] for r in recs} == {"train", "eval"}

    loaded = Transformer.from_checkpoint(tmp_path / "ck" / "t" / "last.pt")
    assert loaded.checkpoint_meta["step"] == 6
    tcfg.max_steps = 8
    t2 = Trainer(mcfg, tcfg)
    t2.resume()
    assert t2.step == 6
    assert t2.train()["steps"] == 8


def test_safetensors_export_roundtrip(tmp_path):
    m = tiny_model()
    size = m.export(tmp_path / "m.safetensors", torch.float32, info={"step": 7})
    loaded = Transformer.from_checkpoint(tmp_path / "m.safetensors").eval()
    x = torch.randint(0, 320, (1, 16))
    torch.testing.assert_close(loaded(x)[0], m(x)[0])
    assert loaded.checkpoint_meta["step"] == 7
    assert loaded.lm_head.weight.data_ptr() == loaded.tok_emb.weight.data_ptr()  # tying survives
    m.export(tmp_path / "h.safetensors", torch.bfloat16)
    assert (tmp_path / "h.safetensors").stat().st_size < 0.6 * size
