"""Byte-level BPE tokenizer, written from scratch.

The design follows GPT-2/GPT-4: text is first split into "chunks" by a regex (so merges
never cross word/number/punctuation boundaries), each chunk is UTF-8 encoded, and BPE
merges are applied to the byte sequence. Because the base alphabet is the 256 byte
values, any string round-trips losslessly - there is no <unk>.

Training (``Tokenizer.train``) works on the table of *unique* chunks with their counts
rather than on the raw text, and maintains pair counts incrementally:

* ``pair_counts[p]``  - frequency-weighted count of adjacent pair ``p``
* ``pair_where[p]``   - indices of the unique words that contain ``p``
* a max-heap of ``(-count, pair)`` with lazy invalidation

Each merge only touches the words that contain the merged pair, so training a 8k vocab
on ~100 MB of text takes about a minute in pure Python instead of hours for the naive
"recount everything after every merge" loop.

``Tokenizer.from_tiktoken`` rebuilds OpenAI's cl100k_base merges from tiktoken's
rank table; the test-suite checks token-for-token parity with tiktoken, which is a
strong end-to-end check of the encoder.
"""

from __future__ import annotations

import argparse
import heapq
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import regex

# The split pattern used by GPT-4 (cl100k_base). Possessive quantifiers need `regex`.
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+"""
    r"""|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)
EOT = "<|endoftext|>"

Pair = tuple[int, int]


def _merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` (left to right) by ``new_id``."""
    out: list[int] = []
    i, n = 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _count_chunks(args: tuple[str, str]) -> Counter[str]:
    text, pattern = args
    return Counter(regex.findall(pattern, text))


class Tokenizer:
    """Byte-level BPE tokenizer.

    Attributes:
        pattern: regex used to pre-split text into chunks.
        merges: ``(left_id, right_id) -> merged_id``. Merged ids increase with merge
            order, so "lowest merged id first" is the BPE priority rule.
        byte_ids: token id of each of the 256 byte values (identity for tokenizers
            trained here; a permutation for tiktoken encodings).
        special_tokens: e.g. ``{"<|endoftext|>": 8191}``.
    """

    def __init__(
        self,
        merges: dict[Pair, int] | None = None,
        pattern: str = GPT4_SPLIT_PATTERN,
        special_tokens: dict[str, int] | None = None,
        byte_ids: list[int] | None = None,
    ):
        self.pattern = pattern
        self.merges: dict[Pair, int] = dict(merges or {})
        self.special_tokens: dict[str, int] = dict(special_tokens or {})
        self.byte_ids: list[int] = list(byte_ids) if byte_ids is not None else list(range(256))
        self._split = regex.compile(pattern)
        self._special_split = None
        if self.special_tokens:
            alt = "|".join(regex.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
            self._special_split = regex.compile(f"({alt})")
        self._cache: dict[str, list[int]] = {}
        self._build_vocab()

    # ------------------------------------------------------------------ vocabulary

    def _build_vocab(self) -> None:
        vocab: dict[int, bytes] = {tid: bytes([b]) for b, tid in enumerate(self.byte_ids)}
        for (a, b), new in sorted(self.merges.items(), key=lambda kv: kv[1]):
            vocab[new] = vocab[a] + vocab[b]
        for s, tid in self.special_tokens.items():
            vocab[tid] = s.encode("utf-8")
        self.vocab = vocab
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}

    @property
    def vocab_size(self) -> int:
        return max(self.vocab) + 1

    @property
    def eot_id(self) -> int:
        return self.special_tokens[EOT]

    def token_bytes(self, tid: int) -> bytes:
        return self.vocab[tid]

    # ------------------------------------------------------------------ training

    @classmethod
    def train(
        cls,
        text: str | Iterable[str],
        vocab_size: int,
        pattern: str = GPT4_SPLIT_PATTERN,
        special_tokens: Iterable[str] = (EOT,),
        workers: int = 1,
        verbose: bool = False,
    ) -> Tokenizer:
        """Learn ``vocab_size - 256 - len(special_tokens)`` merges from ``text``.

        ``text`` may be one string or an iterable of string blocks (counted in parallel
        when ``workers > 1``). Ties between equally frequent pairs are broken by the
        smaller pair, which makes training fully deterministic.
        """
        specials = list(special_tokens)
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError("vocab_size must be at least 256 + number of special tokens")
        t0 = time.time()
        blocks = [text] if isinstance(text, str) else list(text)
        if workers > 1 and len(blocks) > 1:
            with mp.get_context("spawn").Pool(workers) as pool:
                counters = pool.map(_count_chunks, [(b, pattern) for b in blocks])
        else:
            counters = [_count_chunks((b, pattern)) for b in blocks]
        counts: Counter[str] = Counter()
        for c in counters:
            counts.update(c)
        # Special-token text must never be learned as ordinary merges.
        for s in specials:
            counts.pop(s, None)
        if verbose:
            print(f"[bpe] {sum(counts.values()):,} chunks, {len(counts):,} unique "
                  f"({time.time() - t0:.1f}s)")

        words: list[list[int]] = [list(w.encode("utf-8")) for w in counts]
        freqs: list[int] = list(counts.values())
        pair_counts: dict[Pair, int] = defaultdict(int)
        pair_where: dict[Pair, set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            f = freqs[wi]
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                pair_where[p].add(wi)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: dict[Pair, int] = {}
        for k in range(n_merges):
            best = None
            while heap:
                negc, p = heapq.heappop(heap)
                if negc < 0 and pair_counts.get(p, 0) == -negc:
                    best = p
                    break
            if best is None:  # corpus exhausted: every word is a single token
                break
            new_id = 256 + k
            merges[best] = new_id
            touched: set[Pair] = set()
            for wi in pair_where.pop(best, ()):
                w, f = words[wi], freqs[wi]
                if len(w) < 2:
                    continue
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    touched.add(p)
                w = _merge(w, best, new_id)
                words[wi] = w
                for p in zip(w, w[1:]):
                    pair_counts[p] += f
                    pair_where[p].add(wi)
                    touched.add(p)
            pair_counts.pop(best, None)
            for p in touched:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                elif p in pair_counts:
                    del pair_counts[p]
            if verbose and (k + 1) % 1000 == 0:
                print(f"[bpe] merge {k + 1}/{n_merges} ({time.time() - t0:.1f}s)")

        base = 256 + len(merges)
        special_map = {s: base + i for i, s in enumerate(specials)}
        if verbose:
            print(f"[bpe] done: {len(merges)} merges in {time.time() - t0:.1f}s")
        return cls(merges=merges, pattern=pattern, special_tokens=special_map)

    # ------------------------------------------------------------------ encoding

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        bid = self.byte_ids
        ids = [bid[b] for b in chunk.encode("utf-8")]
        merges = self.merges
        while len(ids) >= 2:
            best_rank, best_pair = None, None
            for p in zip(ids, ids[1:]):
                r = merges.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, p
            if best_pair is None:
                break
            ids = _merge(ids, best_pair, best_rank)  # type: ignore[arg-type]
        if len(self._cache) > 500_000:
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as ordinary text."""
        out: list[int] = []
        enc = self._encode_chunk
        for chunk in self._split.findall(text):
            out.extend(enc(chunk))
        return out

    def encode(self, text: str, allowed_special: str | set[str] = "all") -> list[int]:
        """Encode text; special tokens in ``allowed_special`` map to their ids."""
        if allowed_special == "none" or not self._special_split:
            return self.encode_ordinary(text)
        allowed = set(self.special_tokens) if allowed_special == "all" else set(allowed_special)
        out: list[int] = []
        for part in self._special_split.split(text):
            if not part:
                continue
            if part in allowed:
                out.append(self.special_tokens[part])
            else:
                out.extend(self.encode_ordinary(part))
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.vocab[i] for i in ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ interop / io

    @classmethod
    def from_tiktoken(cls, name: str = "cl100k_base") -> Tokenizer:
        """Rebuild a tiktoken encoding's merge list from its byte-string rank table.

        tiktoken stores only ``bytes -> rank``. For each multi-byte token we re-run BPE
        on its own bytes using only lower-ranked merges; the two parts left just before
        the token itself would be formed are exactly its merge pair.
        """
        import tiktoken

        enc = tiktoken.get_encoding(name)
        ranks: dict[bytes, int] = enc._mergeable_ranks
        merges: dict[Pair, int] = {}
        for token, rank in ranks.items():
            if len(token) == 1:
                continue
            left, right = _bpe_split(ranks, token, max_rank=rank)
            merges[(ranks[left], ranks[right])] = rank
        byte_ids = [ranks[bytes([b])] for b in range(256)]
        return cls(merges=merges, pattern=enc._pat_str, special_tokens=dict(enc._special_tokens),
                   byte_ids=byte_ids)

    def save(self, path: str | Path) -> None:
        obj = {
            "format": "forge-bpe-v1",
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "merges": [[a, b, m] for (a, b), m in sorted(self.merges.items(), key=lambda kv: kv[1])],
        }
        if self.byte_ids != list(range(256)):
            obj["byte_ids"] = self.byte_ids
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = {(a, b): m for a, b, m in obj["merges"]}
        return cls(merges=merges, pattern=obj["pattern"], special_tokens=obj["special_tokens"],
                   byte_ids=obj.get("byte_ids"))


def _bpe_split(ranks: dict[bytes, int], token: bytes, max_rank: int) -> tuple[bytes, bytes]:
    parts = [bytes([b]) for b in token]
    while True:
        min_idx, min_rank = -1, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < max_rank and (min_rank is None or r < min_rank):
                min_idx, min_rank = i, r
        if min_rank is None:
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    if len(parts) != 2:
        raise ValueError(f"could not recover a merge pair for {token!r}: {parts}")
    return parts[0], parts[1]


def _read_blocks(path: Path, max_bytes: int, block: int = 8 << 20) -> list[str]:
    blocks, read = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        while read < max_bytes:
            s = f.read(min(block, max_bytes - read))
            if not s:
                break
            blocks.append(s)
            read += len(s)
    return blocks


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    ap.add_argument("--input", type=Path, default=Path("data/tinystories/train.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/tokenizer.json"))
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-mb", type=float, default=100.0, help="train on this many MB of input")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    blocks = _read_blocks(args.input, int(args.max_mb * 1e6))
    # Story separators are not part of the language; drop them before counting.
    blocks = [b.replace(EOT, "\n") for b in blocks]
    tok = Tokenizer.train(blocks, args.vocab_size, workers=args.workers, verbose=True)
    tok.save(args.out)
    sample = "Once upon a time, there was a little girl named Lily."
    ids = tok.encode(sample)
    print(f"[bpe] saved {args.out}; vocab={tok.vocab_size}; '{sample}' -> {len(ids)} tokens")
    print("[bpe] longest tokens:", sorted((tok.vocab[i] for i in tok.vocab), key=len)[-8:])


if __name__ == "__main__":
    sys.exit(main())
