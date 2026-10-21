from __future__ import annotations

from collections import Counter

import pytest
import regex

from forge.tokenizer import EOT, GPT4_SPLIT_PATTERN, Tokenizer, _merge

from .conftest import CORPUS

DIVERSE = [
    "",
    "hello world",
    "Hello, World! 123456789 ... ???",
    "naïve café — déjà vu; 東京タワー; Ελληνικά; русский текст",
    "emoji 🤖🔥👍🏽 and flags 🇯🇵 and zwj 👨‍👩‍👧",
    "def f(x):\n    return x**2  # comment\n\n\tindented\ttabs",
    "   leading spaces and trailing   ",
    "\n\n\nmultiple\r\nline\rbreaks\n",
    "I'm you're they'll we've she'd IT'S DON'T",
    "numbers 3.14159 1,000,000 2024-10-21 0x1f",
]


def naive_bpe_merges(text: str, n_merges: int) -> dict[tuple[int, int], int]:
    """Reference trainer: recount every pair after every merge (slow but obviously right)."""
    words = Counter(regex.findall(GPT4_SPLIT_PATTERN, text))
    seqs = {w: list(w.encode()) for w in words}
    merges = {}
    for k in range(n_merges):
        counts: Counter[tuple[int, int]] = Counter()
        for w, ids in seqs.items():
            for p in zip(ids, ids[1:]):
                counts[p] += words[w]
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))  # max count, ties -> smaller pair
        merges[best] = 256 + k
        seqs = {w: _merge(ids, best, 256 + k) for w, ids in seqs.items()}
    return merges


def test_incremental_trainer_matches_naive_reference():
    n_merges = 120
    tok = Tokenizer.train(CORPUS, vocab_size=256 + n_merges, special_tokens=())
    assert tok.merges == naive_bpe_merges(CORPUS, n_merges)


def test_training_is_deterministic():
    a = Tokenizer.train(CORPUS, vocab_size=300)
    b = Tokenizer.train(CORPUS, vocab_size=300)
    assert a.merges == b.merges and a.special_tokens == b.special_tokens


def test_parallel_counting_matches_serial():
    blocks = [CORPUS[i:i + 1000] for i in range(0, len(CORPUS), 1000)]
    assert Tokenizer.train(blocks, 300, workers=1).merges == Tokenizer.train(blocks, 300, workers=2).merges


@pytest.mark.parametrize("text", DIVERSE)
def test_roundtrip(tiny_tokenizer: Tokenizer, text: str):
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text


def test_merges_compress(tiny_tokenizer: Tokenizer):
    text = CORPUS[:400]
    assert len(tiny_tokenizer.encode(text)) < len(text.encode()) / 2


def test_special_tokens(tiny_tokenizer: Tokenizer):
    eot = tiny_tokenizer.special_tokens[EOT]
    assert eot == tiny_tokenizer.vocab_size - 1
    ids = tiny_tokenizer.encode(f"Lily{EOT}Tom")
    assert ids.count(eot) == 1
    assert tiny_tokenizer.decode(ids) == f"Lily{EOT}Tom"
    assert eot not in tiny_tokenizer.encode(f"Lily{EOT}Tom", allowed_special="none")


def test_save_load(tmp_path, tiny_tokenizer: Tokenizer):
    path = tmp_path / "tok.json"
    tiny_tokenizer.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == tiny_tokenizer.merges
    assert loaded.special_tokens == tiny_tokenizer.special_tokens
    for text in DIVERSE:
        assert loaded.encode(text) == tiny_tokenizer.encode(text)


@pytest.fixture(scope="module")
def cl100k():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # offline CI without the cached BPE file
        pytest.skip(f"cl100k_base unavailable: {e}")
    return enc, Tokenizer.from_tiktoken("cl100k_base")


@pytest.mark.parametrize("text", DIVERSE + [CORPUS[:2000]])
def test_cl100k_parity_with_tiktoken(cl100k, text: str):
    enc, ours = cl100k
    assert ours.encode_ordinary(text) == enc.encode_ordinary(text)
    assert ours.encode(f"{text}{EOT}", allowed_special="all") == enc.encode(f"{text}{EOT}", allowed_special="all")
    assert ours.decode(enc.encode_ordinary(text)) == text
