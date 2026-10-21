"""A Llama-style decoder-only transformer.

Architecture (per block, pre-norm)::

    h   = x + Attn(RMSNorm(x))        # RoPE on q/k, grouped-query attention
    out = h + SwiGLU(RMSNorm(h))

Design notes
------------
* **RMSNorm** (Zhang & Sennrich 2019): no mean-centering, no bias; cheaper than LayerNorm
  and what Llama/Mistral use. Computed in fp32 for stability under bf16 autocast.
* **RoPE** (Su et al. 2021): rotates each (q, k) feature pair by an angle proportional to
  the position, so ``<q_m, k_n>`` depends only on ``m - n``. No learned position table,
  and the KV cache stays valid as the sequence grows.
* **GQA** (Ainslie et al. 2023): ``n_kv_heads < n_heads``; each K/V head is shared by a
  group of query heads. With 8 query heads and 2 KV heads the KV cache is 4x smaller,
  which is what bounds decode throughput and batch size at inference.
* **SwiGLU** (Shazeer 2020): ``W2(silu(W1 x) * W3 x)`` with hidden size ~8/3 d so the
  parameter count matches a 4d GELU MLP. W1/W3 are fused into one matmul.
* Q/K/V projections are fused into one matmul (fewer kernel launches when decoding).
* Weight tying between the embedding and the LM head; residual output projections are
  initialised with std ``0.02 / sqrt(2 * n_layers)`` (GPT-2) so the residual stream's
  variance does not grow with depth.
* Attention uses ``F.scaled_dot_product_attention`` when the installed torch has it
  (fused / flash kernels), otherwise an explicit masked-softmax path. The explicit path
  is also used when attention maps are requested for visualisation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.attention import chunked_causal_attention

HAS_SDPA = hasattr(F, "scaled_dot_product_attention")
USE_CHUNKED_ATTENTION = True  # toggled by benchmarks to compare against the naive path


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_dim: int | None = None
    multiple_of: int = 64
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.ffn_dim is None:
            hidden = int(8 * self.d_model / 3)
            self.ffn_dim = self.multiple_of * math.ceil(hidden / self.multiple_of)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# --------------------------------------------------------------------------- building blocks


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).type_as(x)


def rope_tables(head_dim: int, max_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_len, head_dim) in the "rotate-half" layout."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    t = torch.arange(max_len, dtype=torch.float64)
    freqs = torch.outer(t, inv_freq)  # (max_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float(), emb.sin().float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (B, H, T, D) features; ``cos``/``sin`` are (T, D) for the absolute positions of x."""
    d = x.shape[-1] // 2
    xf = x.float()
    rotated = torch.cat([-xf[..., d:], xf[..., :d]], dim=-1)
    return (xf * cos + rotated * sin).type_as(x)


class KVCache:
    """Preallocated per-layer key/value cache.

    ``length`` is the number of valid positions. Speculative decoding "rolls back" simply
    by lowering ``length``; stale entries past it are overwritten by the next write.
    """

    def __init__(self, cfg: ModelConfig, batch_size: int, max_len: int | None = None,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.max_len = max_len or cfg.max_seq_len
        shape = (cfg.n_layers, batch_size, cfg.n_kv_heads, self.max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = start + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > {self.max_len}")
        self.k[layer, :, :, start:end] = k.to(self.k.dtype)
        self.v[layer, :, :, start:end] = v.to(self.v.dtype)
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wqkv = nn.Linear(cfg.d_model, (cfg.n_heads + 2 * cfg.n_kv_heads) * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, causal: torch.Tensor,
                start: int, cache: KVCache | None, capture: bool) -> torch.Tensor:
        B, T, _ = x.shape
        H, KV, hd = self.n_heads, self.n_kv, self.hd
        q, k, v = self.wqkv(x).split([H * hd, KV * hd, KV * hd], dim=-1)
        q = q.view(B, T, H, hd).transpose(1, 2)
        k = k.view(B, T, KV, hd).transpose(1, 2)
        v = v.view(B, T, KV, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v, start)
            k, v = k.to(q.dtype), v.to(q.dtype)
        Tk = k.shape[2]
        if KV != H:  # GQA: query head h reads kv head h // (H // KV)
            g = H // KV
            k = k[:, :, None].expand(B, KV, g, Tk, hd).reshape(B, H, Tk, hd)
            v = v[:, :, None].expand(B, KV, g, Tk, hd).reshape(B, H, Tk, hd)

        drop = self.dropout if self.training else 0.0
        if HAS_SDPA and not capture:
            if start == 0 and Tk == T:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
            else:
                mask = causal[start:start + T, :Tk]
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        elif (USE_CHUNKED_ATTENTION and not capture and drop == 0 and start == 0 and Tk == T and T > 1
              and torch.is_grad_enabled()):
            # training without fused kernels: O(T) activation memory instead of O(T^2)
            y = chunked_causal_attention(q, k, v, 1.0 / math.sqrt(hd))
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
            if T > 1:
                att = att.masked_fill(~causal[start:start + T, :Tk], float("-inf"))
            att = att.float().softmax(dim=-1)
            if capture:
                self.last_attn = att.detach()
            if drop > 0:
                att = F.dropout(att, p=drop)
            y = att.to(v.dtype) @ v
        y = y.transpose(1, 2).reshape(B, T, H * hd)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.ffn_dim is not None
        self.ffn_dim = cfg.ffn_dim
        self.w13 = nn.Linear(cfg.d_model, 2 * cfg.ffn_dim, bias=False)  # fused gate & up
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).split(self.ffn_dim, dim=-1)
        return self.dropout(self.w2(F.silu(gate) * up))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, causal, start, cache, capture):  # type: ignore[no-untyped-def]
        x = x + self.resid_drop(self.attn(self.attn_norm(x), cos, sin, causal, start, cache, capture))
        return x + self.ffn(self.ffn_norm(x))


# --------------------------------------------------------------------------- model


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        causal = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal", causal, persistent=False)
        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("attn.wo.weight") or name.endswith("ffn.w2.weight"):
                nn.init.normal_(p, mean=0.0, std=resid_std)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear | nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
        start_pos: int | None = None,
        capture_attn: bool = False,
        last_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model on ``idx`` (B, T).

        With a ``cache`` the tokens are placed at positions ``start_pos .. start_pos+T-1``
        (default: ``cache.length``) and the cache is advanced. ``last_only`` computes
        logits only for the final position (prefill for sampling).
        """
        B, T = idx.shape
        start = (cache.length if cache is not None else 0) if start_pos is None else start_pos
        if start + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {start + T} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos = self.rope_cos[start:start + T]
        sin = self.rope_sin[start:start + T]
        x = self.drop(self.tok_emb(idx))
        for layer in self.layers:
            x = layer(x, cos, sin, self.causal, start, cache, capture_attn)
        if cache is not None:
            cache.length = start + T
        x = self.norm(x)
        if last_only and targets is None:
            x = x[:, -1:, :]
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    def attention_maps(self) -> list[torch.Tensor]:
        """Attention probabilities captured by the last ``capture_attn=True`` forward."""
        maps = [blk.attn.last_attn for blk in self.layers]
        if any(m is None for m in maps):
            raise RuntimeError("run forward(..., capture_attn=True) first")
        return maps  # type: ignore[return-value]

    def new_cache(self, batch_size: int = 1, max_len: int | None = None,
                  dtype: torch.dtype | None = None) -> KVCache:
        p = next(self.parameters())
        return KVCache(self.cfg, batch_size, max_len, device=p.device, dtype=dtype or p.dtype)

    # ------------------------------------------------------------------ accounting

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def flops_per_token(self, seq_len: int | None = None) -> float:
        """Training FLOPs per token: 6N for the matmuls + 12*L*d*T for attention (PaLM, App. B)."""
        T = seq_len or self.cfg.max_seq_len
        c = self.cfg
        return 6 * self.num_params(non_embedding=True) + 6 * c.d_model * c.vocab_size + 12 * c.n_layers * c.d_model * T

    def configure_optimizer(self, lr: float, weight_decay: float, betas: tuple[float, float],
                            device_type: str) -> torch.optim.Optimizer:
        """AdamW with weight decay on matrices only (not on norms/gains)."""
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in params.values() if p.dim() >= 2]
        no_decay = [p for p in params.values() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        import inspect

        kwargs: dict[str, Any] = {}
        sig = inspect.signature(torch.optim.AdamW).parameters
        if "fused" in sig and device_type == "cuda":
            kwargs["fused"] = True
        elif "foreach" in sig:
            kwargs["foreach"] = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **kwargs)

    # ------------------------------------------------------------------ io

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: torch.device | str = "cpu",
                        dtype: torch.dtype = torch.float32) -> Transformer:
        """Load a training checkpoint (``.pt``) or an exported weights-only ``.safetensors``."""
        path = Path(path)
        if path.suffix == ".safetensors":
            from safetensors import safe_open
            from safetensors.torch import load_file

            with safe_open(str(path), framework="pt") as f:
                meta = f.metadata() or {}
            config = json.loads(meta["config"])
            state = load_file(str(path))
            info = json.loads(meta.get("info", "{}"))
        else:
            ckpt = torch.load(path, map_location="cpu")
            config, state = ckpt["config"], ckpt["model"]
            info = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer", "rng", "train_config")}
        model = cls(ModelConfig.from_dict(config))
        if "lm_head.weight" not in state and model.cfg.tie_embeddings:
            state["lm_head.weight"] = state["tok_emb.weight"]
        model.load_state_dict({k: v.to(dtype) for k, v in state.items()})
        model.checkpoint_meta = info
        return model.to(device)

    def export(self, path: str | Path, dtype: torch.dtype = torch.bfloat16, info: dict[str, Any] | None = None) -> int:
        """Write inference weights only (no optimizer / RNG) as safetensors; returns the file size."""
        from safetensors.torch import save_file

        state = {k: v.detach().to(dtype).contiguous() for k, v in self.state_dict().items()
                 if not (k == "lm_head.weight" and self.cfg.tie_embeddings)}  # tied: stored once
        meta = {"config": json.dumps(self.cfg.to_dict()), "info": json.dumps(info or {}, default=str)}
        save_file(state, str(path), metadata=meta)
        return Path(path).stat().st_size
