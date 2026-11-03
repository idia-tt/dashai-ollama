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
    "top_p": 0.8,
    "top_k": 20,
    "context_window": 2048,
    "think": "auto",
    "keep_alive": "5m",
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
        "top_p": 0.8,
        "top_k": 20,
        "num_ctx": 2048,
    }
    assert body["keep_alive"] == "5m"
    assert "think" not in body  # "auto" leaves the model's own default


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


def _ok(request):
    return httpx.Response(
        200,
        json={"model": "m", "message": {"role": "assistant", "content": "ok"}},
    )


@pytest.mark.parametrize(
    ("think", "expected"), [("on", True), ("off", False), ("auto", None)]
)
def test_think_setting_is_forwarded(fake_server, think, expected):
    seen, state = fake_server
    state["handler"] = _ok

    OllamaModel(**{**PARAMS, "think": think}).generate(
        [{"role": "user", "content": "x"}]
    )

    assert json.loads(seen[0].content).get("think") is expected


@pytest.mark.parametrize(
    ("raw", "sent"), [("-1", -1), ("0", 0), ("30m", "30m"), ("1h", "1h")]
)
def test_keep_alive_numbers_are_sent_as_numbers(fake_server, raw, sent):
    seen, state = fake_server
    state["handler"] = _ok

    OllamaModel(**{**PARAMS, "keep_alive": raw}).generate(
        [{"role": "user", "content": "x"}]
    )

    assert json.loads(seen[0].content)["keep_alive"] == sent
