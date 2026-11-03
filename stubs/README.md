# DashAI host stubs

`dashai-ollama` is a plugin: it imports `DashAI.*` at runtime, but DashAI is the
host application and not a dependency, so it is not installed in CI.

This folder is a **minimal stand-in** that exposes only the names the plugin
uses, with just enough behaviour for the unit tests. It is put on `pythonpath`
for pytest and on `extra-paths` for ty (see `pyproject.toml`).

It is not a copy of DashAI. If the plugin starts using a new DashAI name, add it
here; if DashAI changes a contract the stubs mirror, update them. Behaviour that
depends on the real host (registration, the UI, a full RAG run) is verified by
hand against a real DashAI install.
