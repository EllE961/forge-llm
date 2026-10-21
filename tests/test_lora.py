from __future__ import annotations

import torch

from forge.lora import LoRALinear, apply_lora, load_adapter, lora_modules, merge_lora, save_adapter, unmerge_lora

from .conftest import tiny_model


def _randomize_b(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for m in lora_modules(model):
            m.lora_B.normal_(0, 0.05)


def test_zero_init_is_identity():
    base = tiny_model()
    x = torch.randint(0, 320, (2, 16))
    ref = base(x)[0]
    apply_lora(base, r=4, alpha=8)
    torch.testing.assert_close(base(x)[0], ref)


def test_trainable_parameter_count_and_freezing():
    m = tiny_model()
    n = apply_lora(m, r=4, alpha=8)
    expected = sum(4 * (lm.base.in_features + lm.base.out_features) for lm in lora_modules(m))
    assert n == expected
    assert len(lora_modules(m)) == 4 * m.cfg.n_layers
    loss = m(torch.randint(0, 320, (2, 16)), torch.randint(0, 320, (2, 16)))[1]
    loss.backward()
    for name, p in m.named_parameters():
        assert (p.grad is not None) == ("lora_" in name), name


def test_merge_matches_unmerged_and_unmerge_restores():
    m = tiny_model()
    apply_lora(m, r=4, alpha=8)
    _randomize_b(m)
    x = torch.randint(0, 320, (2, 16))
    unmerged = m(x)[0]
    w0 = lora_modules(m)[0].base.weight.clone()
    merge_lora(m)
    torch.testing.assert_close(m(x)[0], unmerged, atol=1e-5, rtol=1e-5)
    unmerge_lora(m)
    torch.testing.assert_close(lora_modules(m)[0].base.weight, w0, atol=1e-6, rtol=0)


def test_adapter_roundtrip(tmp_path):
    m = tiny_model()
    apply_lora(m, r=2, alpha=4)
    _randomize_b(m)
    size = save_adapter(m, tmp_path / "a.pt", meta={"r": 2})
    assert size < 200_000  # adapters are tiny compared with the model
    fresh = tiny_model()
    apply_lora(fresh, r=2, alpha=4)
    assert load_adapter(fresh, tmp_path / "a.pt") == {"r": 2}
    x = torch.randint(0, 320, (1, 10))
    torch.testing.assert_close(fresh(x)[0], m(x)[0])


def test_lora_linear_delta():
    lin = torch.nn.Linear(6, 5, bias=False)
    lo = LoRALinear(lin, r=2, alpha=4)
    with torch.no_grad():
        lo.lora_B.normal_()
    x = torch.randn(3, 6)
    torch.testing.assert_close(lo(x), x @ (lin.weight + lo.delta_weight()).T)
