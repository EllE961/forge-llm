"""Datasets: download, parallel tokenization into uint16 memmaps, batch sampling.

The training corpus is TinyStories V2 (GPT-4 generated short stories, Eldan & Li 2023).
Only a prefix of the 2.2 GB training file is fetched (HTTP Range request) and cut at a
story boundary, which keeps the on-disk footprint small while still giving a few hundred
million tokens. Token streams are stored as flat ``uint16`` arrays (vocab < 65536), one
``<|endoftext|>`` token after every story, and read back through ``np.memmap`` so the
training process never holds the corpus in RAM.

CLI::

    python -m forge.data download                 # TinyStories (valid + train prefix) + TinyShakespeare
    python -m forge.data tokenize --tokenizer data/tokenizer.json
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

TINYSTORIES_BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TINYSTORIES_TRAIN = f"{TINYSTORIES_BASE}/TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VALID = f"{TINYSTORIES_BASE}/TinyStoriesV2-GPT4-valid.txt"
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
EOT = "<|endoftext|>"
DATA_DIR = Path("data")


# ----------------------------------------------------------------------------- download


def download(url: str, dest: Path, max_bytes: int | None = None, chunk: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest``; with ``max_bytes`` only that prefix is requested (Range)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[data] {dest} exists ({dest.stat().st_size / 1e6:.1f} MB), skipping")
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "forge-llm/0.1"})
    if max_bytes is not None:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0, done = time.time(), 0
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = max_bytes or int(resp.headers.get("Content-Length", 0)) or None
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            if max_bytes is not None and done + len(buf) > max_bytes:  # server ignored Range
                buf = buf[: max_bytes - done]
            f.write(buf)
            done += len(buf)
            if total:
                rate = done / max(time.time() - t0, 1e-6) / 1e6
                print(f"\r[data] {dest.name}: {done / 1e6:7.1f}/{total / 1e6:.1f} MB "
                      f"({rate:.1f} MB/s)", end="", flush=True)
            if max_bytes is not None and done >= max_bytes:
                break
    print()
    tmp.replace(dest)
    return dest


def cut_at_story_boundary(path: Path) -> None:
    """Truncate a (Range-downloaded) TinyStories prefix after its last complete story."""
    data = path.read_bytes()
    end = data.rfind(EOT.encode())
    if end < 0:
        raise ValueError(f"no {EOT} found in {path}")
    path.write_bytes(data[: end + len(EOT)] + b"\n")


def download_all(data_dir: Path = DATA_DIR, train_mb: int = 600) -> None:
    ts = data_dir / "tinystories"
    valid = download(TINYSTORIES_VALID, ts / "valid.txt")
    train = ts / "train.txt"
    if not train.exists():
        download(TINYSTORIES_TRAIN, train, max_bytes=train_mb * 1_000_000)
        cut_at_story_boundary(train)
    download(SHAKESPEARE_URL, data_dir / "shakespeare" / "input.txt")
    print(f"[data] train {train.stat().st_size / 1e6:.0f} MB, valid {valid.stat().st_size / 1e6:.1f} MB")


# ----------------------------------------------------------------------------- tokenization

_WORKER_TOK = None


def _init_worker(tokenizer_path: str) -> None:
    global _WORKER_TOK
    from forge.tokenizer import Tokenizer

    _WORKER_TOK = Tokenizer.load(tokenizer_path)


def _encode_block(block: str) -> np.ndarray:
    """Encode a block of whole stories; each story is followed by one EOT token."""
    tok = _WORKER_TOK
    assert tok is not None
    eot = tok.special_tokens[EOT]
    out: list[int] = []
    for story in block.split(EOT):
        story = story.strip()
        if story:
            out.extend(tok.encode_ordinary(story))
            out.append(eot)
    return np.asarray(out, dtype=np.uint16)


def iter_story_blocks(path: Path, block_bytes: int = 4 << 20) -> Iterator[str]:
    """Yield ~``block_bytes`` text blocks that always end on a story boundary."""
    carry = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(block_bytes)
            if not chunk:
                break
            text = carry + chunk
            cut = text.rfind(EOT)
            if cut < 0:
                carry = text
                continue
            yield text[: cut + len(EOT)]
            carry = text[cut + len(EOT):]
    if carry.strip():
        yield carry


def tokenize_text_file(src: Path, dst: Path, tokenizer_path: Path, workers: int = 4) -> int:
    """Tokenize a story file into a flat uint16 ``.bin``; returns the number of tokens."""
    t0, n = time.time(), 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = src.stat().st_size
    seen_bytes = 0
    with open(dst, "wb") as f:
        if workers <= 1:
            _init_worker(str(tokenizer_path))
            for arr in map(_encode_block, iter_story_blocks(src)):
                arr.tofile(f)
                n += arr.size
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_init_worker, initargs=(str(tokenizer_path),)) as pool:
                for arr in pool.imap(_encode_block, iter_story_blocks(src), chunksize=1):
                    arr.tofile(f)
                    n += arr.size
                    seen_bytes += 4 << 20
                    pct = min(100.0, 100 * seen_bytes / total_bytes)
                    print(f"\r[data] {src.name}: {n / 1e6:7.1f}M tokens  {pct:5.1f}%  "
                          f"{time.time() - t0:5.0f}s", end="", flush=True)
    dt = time.time() - t0
    print(f"\n[data] {dst} <- {n:,} tokens in {dt:.0f}s ({total_bytes / max(n, 1):.2f} bytes/token)")
    return n


def tokenize_plain_file(src: Path, dst: Path, tokenizer_path: Path) -> int:
    """Tokenize a small plain-text file (no story separators), e.g. TinyShakespeare."""
    from forge.tokenizer import Tokenizer

    tok = Tokenizer.load(tokenizer_path)
    ids = np.asarray(tok.encode_ordinary(src.read_text(encoding="utf-8")), dtype=np.uint16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(dst)
    print(f"[data] {dst} <- {ids.size:,} tokens")
    return int(ids.size)


# ----------------------------------------------------------------------------- sampling


class TokenDataset:
    """Random fixed-length windows over a flat uint16 token memmap."""

    def __init__(self, path: str | Path, seq_len: int):
        self.path = Path(path)
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        if len(self.data) <= seq_len + 1:
            raise ValueError(f"{path} has only {len(self.data)} tokens (< seq_len={seq_len})")

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(
        self, batch_size: int, rng: np.random.Generator, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, y)`` int64 tensors of shape (B, T) where ``y`` is ``x`` shifted by one."""
        starts = rng.integers(0, len(self.data) - self.seq_len - 1, size=batch_size)
        idx = starts[:, None] + np.arange(self.seq_len + 1)[None, :]
        chunk = torch.from_numpy(self.data[idx].astype(np.int64))
        x, y = chunk[:, :-1], chunk[:, 1:]
        if torch.device(device).type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return x, y

    def split(self, frac: float, seq_len: int | None = None) -> tuple[_View, _View]:
        """Deterministic head/tail split (used for TinyShakespeare which has no valid file)."""
        cut = int(len(self.data) * frac)
        sl = seq_len or self.seq_len
        return _View(self.data[:cut], sl), _View(self.data[cut:], sl)


class _View(TokenDataset):
    def __init__(self, data: np.ndarray, seq_len: int):  # noqa: D107 - thin wrapper
        self.path = Path("<view>")
        self.data = data
        self.seq_len = seq_len


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--data-dir", type=Path, default=DATA_DIR)
    d.add_argument("--train-mb", type=int, default=600)
    t = sub.add_parser("tokenize")
    t.add_argument("--data-dir", type=Path, default=DATA_DIR)
    t.add_argument("--tokenizer", type=Path, default=DATA_DIR / "tokenizer.json")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--delete-raw", action="store_true", help="remove train.txt after tokenizing")
    args = ap.parse_args(argv)

    if args.cmd == "download":
        download_all(args.data_dir, args.train_mb)
    elif args.cmd == "tokenize":
        ts = args.data_dir / "tinystories"
        tokenize_text_file(ts / "valid.txt", ts / "valid.bin", args.tokenizer, args.workers)
        tokenize_text_file(ts / "train.txt", ts / "train.bin", args.tokenizer, args.workers)
        tokenize_plain_file(args.data_dir / "shakespeare" / "input.txt",
                            args.data_dir / "shakespeare" / "all.bin", args.tokenizer)
        if args.delete_raw:
            (ts / "train.txt").unlink()
            print("[data] deleted raw train.txt")


if __name__ == "__main__":
    sys.exit(main())
