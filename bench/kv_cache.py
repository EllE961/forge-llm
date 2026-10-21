"""KV cache vs. full recomputation: per-token decode latency as the context grows.

Without a cache, step t re-runs the model over all t tokens (O(t) work per token, O(T^2)
for a sequence); with the cache, each step feeds one token and attends over stored K/V.

    python -m bench.kv_cache --device cuda --tokens 480
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from bench.common import device_info, save, sync
from forge.model import Transformer


@torch.no_grad()
def decode_latencies(model: Transformer, prompt: torch.Tensor, n: int, use_cache: bool) -> list[float]:
    dev = prompt.device
    seq = prompt.clone()
    cache = model.new_cache(1) if use_cache else None
    lat = []
    logits = model(seq, cache=cache, last_only=True)[0][0, -1]
    for _ in range(n):
        sync(dev)
        t0 = time.perf_counter()
        nxt = logits.argmax().view(1, 1)
        seq = torch.cat([seq, nxt], dim=1)
        if use_cache:
            logits = model(nxt, cache=cache, last_only=True)[0][0, -1]
        else:
            logits = model(seq, last_only=True)[0][0, -1]
        sync(dev)
        lat.append(time.perf_counter() - t0)
    return lat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tinystories-30m/model.safetensors")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tokens", type=int, default=480)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = Transformer.from_checkpoint(args.ckpt, dev).eval()
    prompt = torch.randint(300, 8000, (1, 16), generator=torch.Generator().manual_seed(0)).to(dev)
    decode_latencies(model, prompt, 32, True)  # warm-up
    decode_latencies(model, prompt, 32, False)
    out = {"meta": device_info(dev) | {"ckpt": args.ckpt, "prompt_len": 16, "new_tokens": args.tokens}}
    for name, cache in (("cache", True), ("no_cache", False)):
        runs = np.array([decode_latencies(model, prompt, args.tokens, cache) for _ in range(args.repeats)])
        med = np.median(runs, axis=0)
        out[name] = {"latency_ms": (med * 1e3).round(4).tolist(), "tok_per_s": float(args.tokens / med.sum())}
        print(f"[kv] {name:9s}: {out[name]['tok_per_s']:8.1f} tok/s  (last-token latency {med[-1] * 1e3:.2f} ms)")
    out["speedup"] = out["cache"]["tok_per_s"] / out["no_cache"]["tok_per_s"]
    if dev.type == "cuda":  # same cached decode, captured once as a CUDA graph (no per-token launches/syncs)
        from forge.fastpath import StaticDecoder, fast_generate

        dec = StaticDecoder(model, (1,))
        ids = prompt[0].tolist()
        fast_generate(model, ids, 32, decoder=dec)
        times = []
        for _ in range(args.repeats):
            sync(dev)
            t0 = time.perf_counter()
            fast_generate(model, ids, args.tokens, decoder=dec)
            sync(dev)
            times.append(time.perf_counter() - t0)
        out["graph"] = {"tok_per_s": float(args.tokens / np.median(times))}
        out["graph_speedup_vs_eager_cache"] = out["graph"]["tok_per_s"] / out["cache"]["tok_per_s"]
        print(f"[kv] CUDA graph: {out['graph']['tok_per_s']:8.1f} tok/s "
              f"({out['graph_speedup_vs_eager_cache']:.1f}x over eager cached decode)")
    cache_bytes = model.new_cache(1).nbytes()
    out["cache_mb_per_sequence"] = cache_bytes / 1e6
    mha_bytes = cache_bytes * model.cfg.n_heads / model.cfg.n_kv_heads
    out["mha_cache_mb_per_sequence"] = mha_bytes / 1e6
    print(f"[kv] speedup {out['speedup']:.2f}x; KV cache {cache_bytes / 1e6:.1f} MB/seq (GQA) vs "
          f"{mha_bytes / 1e6:.1f} MB/seq if it were MHA")
    save(f"kv_cache_{dev.type}", out)


if __name__ == "__main__":
    main()
