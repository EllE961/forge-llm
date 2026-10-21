"""LoRA: Low-Rank Adaptation (Hu et al. 2021).

A frozen linear layer ``W`` (out x in) is adapted as ``W + (alpha / r) * B @ A`` with
``A`` (r x in) and ``B`` (out x r). ``B`` starts at zero, so the adapted model is
*exactly* the base model at step 0 and training starts from the pretrained function.
Only ``A`` and ``B`` receive gradients: for r=4 on all projections of the 30M model
that is ~0.9% of the parameters, and the optimizer state shrinks accordingly.

After training the update can be folded into ``W`` (``merge``), so inference has zero
extra latency; ``unmerge`` restores the base weights exactly (up to fp rounding).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGETS = ("wqkv", "wo", "w13", "w2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # same init as nn.Linear
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False
        base.weight.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.merged:
            y = y + F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling
        return y

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self.delta_weight().to(self.base.weight.dtype)
            self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, merged={self.merged}"


def apply_lora(model: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0,
               targets: Iterable[str] = DEFAULT_TARGETS) -> int:
    """Wrap every ``nn.Linear`` whose attribute name is in ``targets``; freeze everything else.

    Returns the number of trainable parameters.
    """
    targets = set(targets)
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    if replaced == 0:
        raise ValueError(f"no Linear layers named {sorted(targets)} found")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_modules(model: nn.Module) -> list[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def merge_lora(model: nn.Module) -> None:
    for m in lora_modules(model):
        m.merge()


def unmerge_lora(model: nn.Module) -> None:
    for m in lora_modules(model):
        m.unmerge()


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def save_adapter(model: nn.Module, path: str | Path, meta: dict | None = None) -> int:
    """Save only the adapter weights; returns the file size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": lora_state_dict(model), "meta": meta or {}}, path)
    return path.stat().st_size


def load_adapter(model: nn.Module, path: str | Path) -> dict:
    obj = torch.load(path, map_location="cpu")
    missing = model.load_state_dict(obj["lora"], strict=False)
    if missing.unexpected_keys:
        raise KeyError(f"unexpected adapter keys: {missing.unexpected_keys[:5]}")
    return obj.get("meta", {})
