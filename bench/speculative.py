"""Speculative decoding benchmark: acceptance rate and wall-clock speedup vs. k and temperature.

Prompts are the first 24 tokens of held-out validation stories. For every (k, temperature)
the same prompts are decoded with plain KV-cached sampling and with speculative decoding;
tokens/s are compared on identical token budgets.

    python -m bench.speculative --device cuda
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from bench.common import device_info, save, sync
from forge.generate import generate
from forge.model import Transformer
from forge.speculative import speculative_generate


def prompts_from_valid(path: str, n: int, length: int, eot: int) -> list[list[int]]:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    starts = np.flatnonzero(data[: 2_000_000] == eot)[:n]
    return [data[s + 1: s + 1 + length].astype(int).tolist() for s in starts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="checkpoints/tinystories-30m/model.safetensors")
    ap.add_argument("--draft", default="checkpoints/draft-6m/model.safetensors")
    ap.add_argument("--valid", default="data/tinystories/valid.bin")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--temps", type=float, nargs="*", default=[0.0, 0.8])
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    target = Transformer.from_checkpoint(args.target, dev).eval()
    draft = Transformer.from_checkpoint(args.draft, dev).eval()
    eot = target.cfg.vocab_size - 1
    prompts = prompts_from_valid(args.valid, args.prompts, 24, eot)

    def timed(fn):  # noqa: ANN001, ANN202
        sync(dev)
        t0 = time.perf_counter()
        r = fn()
        sync(dev)
        return time.perf_counter() - t0, r

    engines = ["eager", "graph"] if dev.type == "cuda" else ["eager"]
    t_dec = d_dec = None
    if "graph" in engines:
        from forge.fastpath import StaticDecoder, fast_generate, fast_speculative

        t_dec = StaticDecoder(target, [1] + [k + 1 for k in args.ks])
        d_dec = StaticDecoder(draft, (1,))

    def run_plain(engine: str, p: list[int], temp: float, seed: int) -> list[int]:
        if engine == "graph":
            return fast_generate(target, p, args.tokens, temperature=temp, seed=seed, decoder=t_dec)
        return generate(target, p, args.tokens, temperature=temp, top_p=None, seed=seed)

    def run_spec(engine: str, p: list[int], k: int, temp: float, seed: int) -> tuple[list[int], dict]:
        if engine == "graph":
            out, st = fast_speculative(target, draft, p, args.tokens, k=k, temperature=temp, seed=seed,
                                       decoders=(t_dec, d_dec))
            return out, {"accepted": round(st["acceptance_rate"] * st["rounds"] * k), "drafted": st["rounds"] * k,
                         "rounds": st["rounds"]}
        return speculative_generate(target, draft, p, args.tokens, k=k, temperature=temp, seed=seed)

    rows = []
    for engine in engines:
        run_plain(engine, prompts[0], 0.0, 0)  # warm-up
        run_spec(engine, prompts[0], 4, 0.0, 0)
        for temp in args.temps:
            base_s, base_n, greedy_ref = 0.0, 0, {}
            for i, p in enumerate(prompts):
                s, out = timed(lambda p=p, i=i, t=temp, e=engine: run_plain(e, p, t, i))
                base_s, base_n = base_s + s, base_n + len(out)
                greedy_ref[i] = out
            base_tps = base_n / base_s
            print(f"[spec] {engine} T={temp}: baseline {base_tps:.1f} tok/s")
            for k in args.ks:
                sp_s, sp_n, acc, drafted, rounds, identical = 0.0, 0, 0, 0, 0, True
                for i, p in enumerate(prompts):
                    s, (out, st) = timed(lambda p=p, i=i, k=k, t=temp, e=engine: run_spec(e, p, k, t, i))
                    sp_s += s
                    sp_n += len(out)
                    acc += st["accepted"]
                    drafted += st["drafted"]
                    rounds += st["rounds"]
                    if temp == 0:
                        identical &= out == greedy_ref[i]
                row = {"engine": engine, "temperature": temp, "k": k, "acceptance_rate": acc / max(drafted, 1),
                       "tokens_per_round": sp_n / max(rounds, 1), "tok_per_s": sp_n / sp_s,
                       "baseline_tok_per_s": base_tps, "speedup": (sp_n / sp_s) / base_tps}
                if temp == 0:
                    row["identical_to_greedy"] = bool(identical)
                rows.append(row)
                print(f"[spec] {engine} T={temp} k={k}: alpha={row['acceptance_rate']:.3f} "
                      f"tok/round={row['tokens_per_round']:.2f} {row['tok_per_s']:.1f} tok/s "
                      f"speedup={row['speedup']:.2f}x" + (f" identical={identical}" if temp == 0 else ""))
    meta = device_info(dev) | {"target_params": target.num_params(), "draft_params": draft.num_params(),
                               "prompts": len(prompts), "tokens_per_prompt": args.tokens}
    save(f"speculative_{dev.type}", {"meta": meta, "rows": rows})


if __name__ == "__main__":
    main()
