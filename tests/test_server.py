from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from forge.server import Playground, create_app

from .conftest import asgi_request, tiny_model


@pytest.fixture(scope="module")
def app(tiny_tokenizer, tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    run = runs / "demo"
    run.mkdir()
    recs = [{"type": "train", "step": s, "tokens": s * 100, "loss": 5 - s / 10, "lr": 1e-3, "grad_norm": 1.0,
             "tok_per_s": 1e4, "mfu": 0.1, "elapsed": s} for s in range(1, 30)]
    recs.append({"type": "eval", "step": 29, "tokens": 2900, "val_loss": 2.5, "val_ppl": 12.2, "elapsed": 29})
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    pg = Playground(tiny_model(vocab_size=320), tiny_tokenizer, draft=tiny_model(vocab_size=320, seed=3, n_layers=1),
                    runs_dir=runs)
    return create_app(pg)


def _sse(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_status(app):
    r = asgi_request(app, "GET", "/api/status").json()
    assert r["model"]["params"] > 0 and r["draft"]["params"] > 0 and r["vocab_size"] == 320


def test_generate_stream(app):
    r = asgi_request(app, "POST", "/api/generate", json={"prompt": "Lily", "max_new_tokens": 12, "seed": 1})
    events = _sse(r.text)
    assert events[0]["type"] == "prompt" and events[-1]["type"] == "done"
    toks = [e for e in events if e["type"] == "token"]
    assert 1 <= len(toks) <= 12
    assert all(0 < t["prob"] <= 1 and len(t["top"]) == 5 for t in toks)


def test_race_greedy_is_identical(app):
    r = asgi_request(app, "POST", "/api/race", json={"prompt": "Tom", "max_new_tokens": 30, "k": 3, "temperature": 0})
    body = r.json()
    assert body["stats"]["identical"] is True
    assert body["speculative"]["tokens"] == len(body["baseline"]["tokens"])
    assert 0 <= body["stats"]["acceptance_rate"] <= 1


def test_attention_payload(app):
    body = asgi_request(app, "POST", "/api/attention", json={"text": "Lily had a red ball."}).json()
    raw = np.frombuffer(base64.b64decode(body["data"]), dtype=np.uint8)
    L, H, T = body["n_layers"], body["n_heads"], body["T"]
    assert raw.size == L * H * T * T
    maps = (raw.reshape(L, H, T, T) / 255.0) ** 2
    assert np.allclose(maps.sum(-1), 1.0, atol=0.08)  # rows ~ distributions after 8-bit quantisation


def test_heads_tokenize_embeddings_neighbors(app):
    heads = asgi_request(app, "GET", "/api/heads").json()
    assert len(heads["heads"]) == heads["n_layers"] * heads["n_heads"]
    tok = asgi_request(app, "POST", "/api/tokenize", json={"text": "Once upon a time"}).json()
    assert tok["n_tokens"] == len(tok["tokens"]) and tok["n_bytes"] == 16
    emb = asgi_request(app, "GET", "/api/embeddings").json()
    assert len(emb["x"]) == len(emb["tokens"]) == 320
    nb = asgi_request(app, "GET", "/api/neighbors/5?k=4").json()
    assert len(nb["neighbors"]) == 4 and all(n["id"] != 5 for n in nb["neighbors"])


def test_runs(app):
    runs = asgi_request(app, "GET", "/api/runs").json()
    assert runs[0]["name"] == "demo" and runs[0]["active"] is True
    detail = asgi_request(app, "GET", "/api/runs/demo?max_points=10").json()
    assert len(detail["train"]) <= 11 and detail["eval"][0]["val_loss"] == 2.5
    assert asgi_request(app, "GET", "/api/runs/..%2Fsecret").status_code == 404


def test_missing_model_returns_503(tiny_tokenizer):
    app = create_app(Playground(None, tiny_tokenizer))
    assert asgi_request(app, "POST", "/api/generate", json={"prompt": "x"}).status_code == 503
    assert asgi_request(app, "GET", "/api/status").json()["model"] is None
