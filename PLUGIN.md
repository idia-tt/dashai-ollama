<!-- This is the long description shown in DashAI's Plugins page and on PyPI.
DashAI renders it with plain CommonMark (react-markdown without GFM): no tables, no
badges, no relative links, no anchors. Keep it to headings, lists, code and absolute
links. The full documentation with tables lives in README.md. -->

## Run chat, RAG and embeddings with Ollama

Use any model served by a local [Ollama](https://ollama.com) server from DashAI. The plugin
does not load weights itself: Ollama owns the models, the GPU and the memory, and DashAI
talks to it over HTTP. Any model from the [Ollama library](https://ollama.com/library) works
without waiting for DashAI to add it.

It adds two components:

- **Ollama Model**: for chat sessions and as the generation model of a RAG pipeline.
- **Ollama Embedding**: as the embedding model of a dense retriever in a RAG pipeline.

## Quickstart

1. Install and start Ollama: <https://ollama.com/download>.
2. Download a chat model and an embedding model:

   ```bash
   ollama pull bonsai-2:9b
   ollama pull nomic-embed-text
   ```

3. Install DashAI with this plugin and start it:

   ```bash
   uv tool install dashai --with dashai-ollama
   dashai
   ```

4. Open <http://localhost:8000>, go to **Generative**, create a session and pick
   **Ollama Model**. The defaults already point at a local Ollama and at `bonsai-2:9b`.

## Configuration

- **Server URL**: where Ollama runs, `http://localhost:11434` by default.
- **Model**: the exact name with its tag, as `ollama list` shows it. Pull it first.
- **Max tokens, Temperature, Top P, Top K**: sampling controls. Reasoning models also spend
  tokens thinking, so give them 4,096 or more.
- **Context window**: tokens kept in memory, 16,384 by default.
- **Thinking**: `auto`, `on` or `off`, only for models that can think.
- **Keep alive**: how long Ollama keeps the model loaded after the last request.

Each field has its own description in the DashAI form. The full parameter reference,
recommended models and troubleshooting are in the README.

## Links

- Source code, documentation and issues: <https://github.com/idia-tt/dashai-ollama>
- Full README: <https://github.com/idia-tt/dashai-ollama#readme>
- Changelog: <https://github.com/idia-tt/dashai-ollama/blob/main/CHANGELOG.md>
- DashAI: <https://github.com/DashAISoftware/DashAI>

Requires DashAI 0.10 or newer and Python 3.11 to 3.14. MIT license.
