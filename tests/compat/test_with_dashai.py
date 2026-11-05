"""Smoke test against a real DashAI install.

The regular tests run against ``stubs/``, which can drift from the host. This
file catches that drift: it needs the real DashAI, so it is deselected by
default and run by the scheduled workflow:

    pytest -m dashai -o pythonpath=src -o addopts=""
"""

import pathlib

import pytest

pytestmark = pytest.mark.dashai

COMPONENTS = ["OllamaModel", "OllamaEmbedding"]


@pytest.fixture(scope="module")
def registry():
    import DashAI

    if "stubs" in pathlib.Path(DashAI.__file__).parts:
        pytest.fail("This test needs the real DashAI, but it imported stubs/.")

    from DashAI.back.dependencies.registry.component_registry import ComponentRegistry

    from dashai_ollama.ollama_embedding import OllamaEmbedding
    from dashai_ollama.ollama_model import OllamaModel

    components = ComponentRegistry()
    for component in (OllamaModel, OllamaEmbedding):
        components.register_component(component)
    return components


@pytest.mark.parametrize("name", COMPONENTS)
def test_component_registers(registry, name):
    assert name in registry


def test_embedding_is_listed_as_a_dense_embedding(registry):
    children = registry.get_child_components("DenseEmbedding", recursive=True)

    assert "OllamaEmbedding" in {child["name"] for child in children}


def test_model_works_for_chat_and_rag(registry):
    component = registry["OllamaModel"]["class"]

    assert {"TextToTextGenerationTask", "RAGTask"} <= set(
        component.COMPATIBLE_COMPONENTS
    )


@pytest.mark.parametrize("name", COMPONENTS)
def test_default_parameters_build_the_component(registry, name):
    component = registry[name]["class"]
    properties = component.get_schema()["properties"]
    defaults = {key: spec["placeholder"] for key, spec in properties.items()}

    # Building the client makes no network call, so no server is needed.
    component(**defaults)


def test_typed_generation_works_on_hosts_that_have_it():
    """``generate_text`` calls ``generate_result`` once and keeps reasoning."""
    generation = pytest.importorskip(
        "DashAI.back.core.generation",
        reason="This DashAI predates the typed text-generation interface.",
    )
    import httpx
    import ollama

    from dashai_ollama.ollama_model import OllamaModel

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "model": "m",
                "message": {"role": "assistant", "content": "A", "thinking": "T"},
                "done_reason": "stop",
                "prompt_eval_count": 3,
                "eval_count": 4,
            },
        )

    real_client = ollama.Client
    ollama.Client = lambda **kw: real_client(
        transport=httpx.MockTransport(handler), **kw
    )
    try:
        defaults = {
            k: v["placeholder"]
            for k, v in OllamaModel.get_schema()["properties"].items()
        }
        model = OllamaModel(**defaults)
    finally:
        ollama.Client = real_client

    result = generation.generate_text(model, [{"role": "user", "content": "x"}])

    assert len(calls) == 1
    assert result.text == "A"
    assert [p.type for p in result.parts] == ["reasoning", "text"]
    assert result.usage.total_tokens == 7
    assert result.finish_reason == "stop"
