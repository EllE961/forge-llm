"""Exact validation loss over the *whole* held-out set (non-overlapping 512-token windows),
plus a fixed set of sample completions for the README.

    python -m bench.eval_val --ckpt checkpoints/tinystories-30m/model.safetensors
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from bench.common import device_info, save
from forge.generate import generate
from forge.model import Transformer
from forge.tokenizer import Tokenizer

PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "Tom and his dog Max went to the park. Suddenly,",
    "The little robot was sad because",
    "Sara found a strange key under her bed.",
]


@torch.no_grad()
def full_val_loss(model: Transformer, path: str, seq_len: int, batch: int, dev: torch.device) -> tuple[float, int]:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n_win = (len(data) - 1) // seq_len
    total, count = 0.0, 0
    for b in range(0, n_win, batch):
        idx = np.arange(b, min(b + batch, n_win))
        rows = np.stack([data[i * seq_len: i * seq_len + seq_len + 1] for i in idx]).astype(np.int64)
        x = torch.from_numpy(rows[:, :-1]).to(dev)
        y = torch.from_numpy(rows[:, 1:]).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total += loss.item()
        count += y.numel()
    return total / count, count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tinystories-30m/model.safetensors")
    ap.add_argument("--valid", default="data/tinystories/valid.bin")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--name", default="eval_main")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model = Transformer.from_checkpoint(args.ckpt, dev).eval()
    tok = Tokenizer.load(args.tokenizer)
    loss, n = full_val_loss(model, args.valid, model.cfg.max_seq_len, 32, dev)
    print(f"[eval] {args.ckpt}: val loss {loss:.4f} (ppl {math.exp(loss):.3f}) over {n:,} tokens")
    samples = []
    for i, p in enumerate(PROMPTS):
        ids = generate(model, tok.encode(p), 160, temperature=0.8, top_p=0.95, seed=i, stop_token=tok.eot_id)
        samples.append({"prompt": p, "completion": tok.decode(ids)})
        print(f"\n[{p}]{tok.decode(ids)}")
    # bits per byte: tokenizer-independent, comparable across vocabularies
    lens = np.array([0 if i == tok.eot_id else len(tok.token_bytes(i)) for i in range(tok.vocab_size)])
    val = np.memmap(args.valid, dtype=np.uint16, mode="r")
    n_bytes = int(lens[val[1: n + 1]].sum())
    bpb = loss * n / math.log(2) / n_bytes
    print(f"[eval] {bpb:.4f} bits/byte")
    meta = getattr(model, "checkpoint_meta", {})
    save(args.name, {"meta": device_info(dev) | {"ckpt": args.ckpt, "step": meta.get("step"),
                                                 "tokens_seen": meta.get("tokens_seen")},
                     "val_loss": loss, "val_ppl": math.exp(loss), "val_tokens": n,
                     "bits_per_byte": bpb,
                     "params": model.num_params(), "samples": samples})


if __name__ == "__main__":
    main()
