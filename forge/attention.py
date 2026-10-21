"""Memory-efficient causal attention as a custom autograd function.

The naive attention keeps the full (B, H, T, T) score and probability tensors alive for
the backward pass. With T=512, 8 heads and a micro-batch of 16 that is ~400 MB *per
layer*, which dominated activation memory on this project's 10 GB consumer GPU (torch
1.13 has no fused ``scaled_dot_product_attention``).

This implements the FlashAttention recipe (Dao et al. 2022) at the PyTorch level:

* forward: process queries in blocks; for each block compute scores against the keys
  it may attend to (causal => only keys ``<= block end``, which also halves the FLOPs),
  and keep just the output ``O`` and the per-row log-sum-exp ``L``.
* backward: recompute ``P = exp(S - L)`` block by block from Q and K and use

      dV = P^T dO,   dP = dO V^T,   dS = P * (dP - rowsum(dO * O)),
      dQ = dS K * scale,   dK = dS^T Q * scale

Saved activations drop from O(T^2) to O(T) per head; the price is recomputing QK^T in
the backward pass (a few percent of total model FLOPs for this size). It is not a fused
kernel - there is no SRAM tiling - but it captures the algorithmic idea and made a 2x
larger micro-batch fit.
"""

from __future__ import annotations

import torch


def _acc_dtype(dt: torch.dtype) -> torch.dtype:
    """Softmax/accumulation precision: fp32 for half types, otherwise the input dtype."""
    return torch.float64 if dt == torch.float64 else torch.float32


def _block_mask(n_rows: int, n_cols: int, offset: int, device: torch.device) -> torch.Tensor:
    # row r (absolute position offset + r) may see columns c <= offset + r
    return torch.ones(n_rows, n_cols, dtype=torch.bool, device=device).tril(diagonal=offset)


class ChunkedCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,  # type: ignore[override]
                block: int) -> torch.Tensor:
        B, H, T, _ = q.shape
        acc = _acc_dtype(q.dtype)
        o = torch.empty_like(q)
        lse = torch.empty(B, H, T, device=q.device, dtype=acc)
        for i0 in range(0, T, block):
            i1 = min(i0 + block, T)
            s = torch.matmul(q[:, :, i0:i1], k[:, :, :i1].transpose(-2, -1)).to(acc) * scale
            s.masked_fill_(~_block_mask(i1 - i0, i1, i0, q.device), float("-inf"))
            row_lse = torch.logsumexp(s, dim=-1, keepdim=True)
            p = torch.exp(s - row_lse)
            o[:, :, i0:i1] = torch.matmul(p.to(v.dtype), v[:, :, :i1])
            lse[:, :, i0:i1] = row_lse.squeeze(-1)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.scale, ctx.block = scale, block
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):  # type: ignore[override]
        q, k, v, o, lse = ctx.saved_tensors
        scale, block = ctx.scale, ctx.block
        B, H, T, _ = q.shape
        acc = _acc_dtype(q.dtype)
        do = do.contiguous()
        delta = (do.to(acc) * o.to(acc)).sum(-1)  # rowsum(dO * O), (B, H, T)
        dq = torch.zeros(q.shape, device=q.device, dtype=acc)
        dk = torch.zeros(k.shape, device=k.device, dtype=acc)
        dv = torch.zeros(v.shape, device=v.device, dtype=acc)
        for i0 in range(0, T, block):
            i1 = min(i0 + block, T)
            qb, dob = q[:, :, i0:i1], do[:, :, i0:i1]
            kk, vv = k[:, :, :i1], v[:, :, :i1]
            s = torch.matmul(qb, kk.transpose(-2, -1)).to(acc) * scale
            s.masked_fill_(~_block_mask(i1 - i0, i1, i0, q.device), float("-inf"))
            p = torch.exp(s - lse[:, :, i0:i1, None])
            dv[:, :, :i1] += torch.matmul(p.to(do.dtype).transpose(-2, -1), dob).to(acc)
            dp = torch.matmul(dob, vv.transpose(-2, -1)).to(acc)
            ds = (p * (dp - delta[:, :, i0:i1, None])).to(q.dtype)
            dq[:, :, i0:i1] = torch.matmul(ds, kk).to(acc) * scale
            dk[:, :, :i1] += torch.matmul(ds.transpose(-2, -1), qb).to(acc) * scale
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def chunked_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                             block: int = 128) -> torch.Tensor:
    """Causal softmax attention over (B, H, T, D) tensors with O(T) saved activations."""
    return ChunkedCausalAttention.apply(q, k, v, scale, block)
