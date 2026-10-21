"""Export a training checkpoint to a weights-only safetensors file for inference.

Training checkpoints carry the AdamW moments (2x the weights) and RNG state; the exported
file is bf16 weights + config (tied LM head stored once): ~54 MB instead of ~320 MB.

    python -m forge.export checkpoints/tinystories-30m/best.pt   # -> .../model.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from forge.model import Transformer


def export(src: str | Path, dst: str | Path | None = None, dtype: torch.dtype = torch.bfloat16) -> Path:
    src = Path(src)
    dst = Path(dst) if dst else src.with_name("model.safetensors")
    model = Transformer.from_checkpoint(src)
    size = model.export(dst, dtype, info=getattr(model, "checkpoint_meta", {}))
    print(f"[export] {src} ({src.stat().st_size / 1e6:.0f} MB) -> {dst} ({size / 1e6:.1f} MB, {dtype})")
    return dst


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = ap.parse_args(argv)
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    export(args.checkpoint, args.out, dtypes[args.dtype])


if __name__ == "__main__":
    sys.exit(main())
