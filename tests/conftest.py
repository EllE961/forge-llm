from __future__ import annotations

import asyncio

import httpx
import pytest
import torch

from forge.model import ModelConfig, Transformer
from forge.tokenizer import Tokenizer

CORPUS = (
    "Once upon a time, there was a little girl named Lily. She loved to play outside in the park. "
    "One day, she saw a big red ball. \"Can I play with you?\" asked Tom. Lily smiled and said yes! "
    "They played all day and became best friends. The end.\n"
) * 20


@pytest.fixture(scope="session")
def tiny_tokenizer() -> Tokenizer:
    return Tokenizer.train(CORPUS, vocab_size=320)


def tiny_model(vocab_size: int = 320, seed: int = 0, **kw) -> Transformer:
    torch.manual_seed(seed)
    cfg = dict(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128)
    cfg.update(kw)
    return Transformer(ModelConfig(**cfg)).eval()


def asgi_request(app, method: str, url: str, **kw) -> httpx.Response:
    """Call an ASGI app in-process (works with any httpx version, unlike older TestClient)."""

    async def go() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kw)

    return asyncio.run(go())
