"""Tests for the Ollama embedding component (no real server needed)."""

import json

import httpx
import numpy as np
import ollama
import pytest

from dashai_ollama.ollama_embedding import OllamaEmbedding

PARAMS = {
    "host": "http://ollama.test:11434",
    "model": "nomic-embed-text",
    "batch_size": 2,
    "timeout": 5,
}
DIM = 4


@pytest.fixture
def fake_server(monkeypatch):
    """Route ollama.Client through an in-memory handler; yields requests seen."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        inputs = seen[-1]["input"]
        return httpx.Response(
            200,
            json={
                "model": "nomic-embed-text",
                "embeddings": [[float(len(t))] * DIM for t in inputs],
            },
        )

    real_client = ollama.Client
    monkeypatch.setattr(
        ollama,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    return seen


def test_encode_returns_float32_vector(fake_server):
    vector = OllamaEmbedding(**PARAMS).encode("abc")

    assert vector.shape == (DIM,)
    assert vector.dtype == np.float32
    assert fake_server[0]["model"] == "nomic-embed-text"


def test_batch_encode_splits_requests_and_keeps_order(fake_server):
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    result = OllamaEmbedding(**PARAMS).batch_encode(texts)

    assert [len(r["input"]) for r in fake_server] == [2, 2, 1]
    assert result.shape == (5, DIM)
    assert result[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_batch_encode_of_nothing_makes_no_request(fake_server):
    assert OllamaEmbedding(**PARAMS).batch_encode([]).shape == (0, 0)
    assert fake_server == []


def test_load_learns_the_embedding_dimension(fake_server):
    embedding = OllamaEmbedding(**PARAMS)

    embedding.load()

    assert embedding.embedding_dim == DIM


def test_unreachable_server_raises_runtime_error(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused")

    real_client = ollama.Client
    monkeypatch.setattr(
        ollama,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(refuse), **kw),
    )

    with pytest.raises(RuntimeError, match="ollama.test"):
        OllamaEmbedding(**PARAMS).load()


def test_missing_model_raises_runtime_error(monkeypatch):
    real_client = ollama.Client
    monkeypatch.setattr(
        ollama,
        "Client",
        lambda **kw: real_client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(404, json={"error": "not found"})
            ),
            **kw,
        ),
    )

    with pytest.raises(RuntimeError, match="nomic-embed-text"):
        OllamaEmbedding(**PARAMS).load()
