"""Tests for the Ollama text-generation component (no real server needed)."""

import json

import httpx
import ollama
import pytest

from dashai_ollama.ollama_model import OllamaModel

PARAMS = {
    "host": "http://ollama.test:11434",
    "model": "llama3.2:3b",
    "max_tokens": 64,
    "temperature": 0.2,
    "context_window": 2048,
    "timeout": 5,
}


@pytest.fixture
def fake_server(monkeypatch):
    """Route ollama.Client through an in-memory handler; yields captured requests."""
    seen = []
    state = {"handler": None}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return state["handler"](request)

    real_client = ollama.Client
    monkeypatch.setattr(
        ollama,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    return seen, state


def test_generate_sends_chat_request_and_returns_reply(fake_server):
    seen, state = fake_server
    state["handler"] = lambda r: httpx.Response(
        200,
        json={
            "model": "llama3.2:3b",
            "message": {"role": "assistant", "content": "hi"},
        },
    )
    messages = [{"role": "user", "content": "hello"}]

    assert OllamaModel(**PARAMS).generate(messages) == ["hi"]

    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/chat"
    assert body["model"] == "llama3.2:3b"
    assert body["messages"] == messages
    assert body["options"] == {
        "num_predict": 64,
        "temperature": 0.2,
        "num_ctx": 2048,
    }


def test_missing_model_raises_runtime_error(fake_server):
    _, state = fake_server
    state["handler"] = lambda r: httpx.Response(404, json={"error": "not found"})

    with pytest.raises(RuntimeError, match="llama3.2:3b"):
        OllamaModel(**PARAMS).generate([{"role": "user", "content": "x"}])


def test_unreachable_server_raises_runtime_error(fake_server):
    _, state = fake_server

    def refuse(request):
        raise httpx.ConnectError("refused")

    state["handler"] = refuse

    with pytest.raises(RuntimeError, match="ollama.test"):
        OllamaModel(**PARAMS).generate([{"role": "user", "content": "x"}])


def test_compatible_with_chat_and_rag():
    assert {"TextToTextGenerationTask", "RAGTask"} <= set(
        OllamaModel.COMPATIBLE_COMPONENTS
    )
    assert OllamaModel.get_metadata()["requires_download"] is False


def test_truncated_reasoning_reply_raises(fake_server):
    _, state = fake_server
    state["handler"] = lambda r: httpx.Response(
        200,
        json={
            "model": "llama3.2:3b",
            "message": {"role": "assistant", "content": "", "thinking": "hmm"},
            "done_reason": "length",
        },
    )

    with pytest.raises(RuntimeError, match="max_tokens"):
        OllamaModel(**PARAMS).generate([{"role": "user", "content": "x"}])
