"""Training step: naive masked-softmax attention vs. the chunked O(T)-memory attention.

Alternates the two implementations so GPU contention affects both equally, and reports
tokens/s and peak allocated memory per micro-batch size.

    python -m bench.attention_memory --batch-sizes 8 16
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import forge.model as fm
from bench.common import Timer, device_info, save
from forge.model import ModelConfig, Transformer


def step_stats(chunked: bool, bs: int, steps: int, dev: torch.device) -> dict[str, float]:
    fm.USE_CHUNKED_ATTENTION = chunked
    torch.manual_seed(0)
    model = Transformer(ModelConfig()).to(dev)
    opt = model.configure_optimizer(1e-3, 0.1, (0.9, 0.95), dev.type)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.integers(0, 8192, (bs, 513))).to(dev)
    torch.cuda.reset_peak_memory_stats(dev)

    def one() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x[:, :-1], x[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):
        one()
    with Timer(dev) as t:
        for _ in range(steps):
            one()
    return {"tok_per_s": steps * bs * 512 / t.seconds, "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 1e9}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[8, 16])
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    dev = torch.device("cuda")
    rows = []
    for bs in args.batch_sizes:
        acc: dict[bool, list[dict[str, float]]] = {True: [], False: []}
        for _ in range(args.rounds):
            for chunked in (False, True):
                acc[chunked].append(step_stats(chunked, bs, args.steps, dev))
                torch.cuda.empty_cache()  # the model/optimizer of the finished run are garbage now
        for chunked, runs in acc.items():
            row = {"batch_size": bs, "attention": "chunked" if chunked else "naive",
                   "tok_per_s": float(np.median([r["tok_per_s"] for r in runs])),
                   "peak_mem_gb": max(r["peak_mem_gb"] for r in runs)}
            rows.append(row)
            print(f"[attn] bs={bs:3d} {row['attention']:8s} {row['tok_per_s'] / 1e3:6.1f}k tok/s  "
                  f"peak {row['peak_mem_gb']:.2f} GB")
    save("attention_memory", {"meta": device_info(dev) | {"seq_len": 512, "model": "27M"}, "rows": rows})


if __name__ == "__main__":
    main()
