# dashai-ollama

[![CI](https://github.com/idia-tt/dashai-ollama/actions/workflows/ci.yml/badge.svg)](https://github.com/idia-tt/dashai-ollama/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A [DashAI](https://github.com/DashAISoftware/DashAI) plugin that runs **chat**, **RAG** and
**embeddings** against a local [Ollama](https://ollama.com) server.

The plugin does not load model weights itself. Ollama owns the models, the GPU and the
memory; DashAI talks to it over HTTP. That keeps DashAI light, and lets you use any model
from the [Ollama library](https://ollama.com/library) without waiting for it to be added to
DashAI.

| Component         | Shown in DashAI as   | Use it for                                                     |
| ----------------- | -------------------- | -------------------------------------------------------------- |
| `OllamaModel`     | **Ollama Model**     | Chat sessions and the generation model of a RAG pipeline       |
| `OllamaEmbedding` | **Ollama Embedding** | The embedding model of any dense retriever in a RAG pipeline   |

## Contents

- [Quickstart](#quickstart)
- [Compatibility](#compatibility)
- [Installing DashAI](#installing-dashai)
- [Using the plugin in DashAI](#using-the-plugin-in-dashai)
- [Parameters](#parameters)
- [Where to find the values](#where-to-find-the-values)
- [Recommended models](#recommended-models)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Conventions](#conventions)
- [Releases](#releases)
- [License](#license)

## Quickstart

From zero to a first answer in four steps. Each one is explained in more detail below.

```bash
# 1. Install and start Ollama (Linux; macOS and Windows: https://ollama.com/download)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Download a chat model and an embedding model
ollama pull bonsai-2:9b
ollama pull nomic-embed-text

# 3. Install DashAI with this plugin and start it
uv tool install dashai --with dashai-ollama
dashai

# 4. Open http://localhost:8000, go to Generative, create a session and pick "Ollama Model"
```

### 1. Install and start Ollama

- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`
- **macOS and Windows:** download the installer from <https://ollama.com/download>.

The installers start the server for you and keep it running in the background. If yours did
not, or you installed Ollama another way, start it by hand and leave that terminal open:

```bash
ollama serve
```

Check that it answers (it listens on `http://localhost:11434` by default):

```bash
curl http://localhost:11434/api/version
```

### 2. Download models

`bonsai-2:9b` is the chat model used while developing this plugin. It is a 9B-parameter
model with ternary weights, small enough (about 3 GB on disk) to run on a laptop, with
reasoning (`thinking`), tool and vision support and a 262,144-token context. It needs a
recent Ollama (**0.40.6 or newer**, check with `ollama --version`).

```bash
ollama pull bonsai-2:9b        # chat and RAG generation
ollama pull nomic-embed-text   # embeddings, only needed for RAG
```

See what you have, and what a model can do:

```bash
ollama list
ollama show bonsai-2:9b
```

`bonsai-2:9b` thinks before it answers. If its replies come back empty or cut off, raise
**Max tokens** or set **Thinking** to *Off* (see [Troubleshooting](#troubleshooting)). To pick
other models, see [Recommended models](#recommended-models).

### 3. Install DashAI and the plugin

See [Installing DashAI](#installing-dashai) for every option (uv, pip, desktop installers).
The shortest one installs both in an isolated environment and puts `dashai` on your PATH:

```bash
uv tool install dashai --with dashai-ollama
dashai
```

Then open <http://localhost:8000>.

### 4. Connect it to DashAI

In the DashAI interface go to **Generative**, create a new session, choose **Ollama Model** and
fill in the form. The defaults already point at a local Ollama and at `bonsai-2:9b`, so for a
first try you can accept them. Details are in [Using the plugin in DashAI](#using-the-plugin-in-dashai).

## Compatibility

| | Supported |
| --- | --- |
| **DashAI** | 0.10 or newer. The embedding component needs the RAG module, which DashAI 0.9 does not have. |
| **Python** | 3.11 to 3.14, tested in CI. |
| **Ollama** | A recent server with the `/api/chat` and `/api/embed` endpoints. Some models need a newer Ollama than others (`bonsai-2` needs 0.40.6). |

The plugin's own dependencies keep only lower bounds (`ollama>=0.6.3`, `httpx>=0.27`,
`numpy>=1.26`), and it does not depend on DashAI, PyTorch or anything else DashAI already
ships. DashAI installs only what its environment is missing, and the copies DashAI ships
always win, so the plugin adds as little as possible and never pins DashAI's libraries.

Three CI checks keep this true:

- **Minimum dependencies** (every pull request): the tests run with the lowest allowed
  versions of the dependencies.
- **Latest dependencies** (weekly): the type check and the tests run without the lock file.
- **DashAI compatibility** (weekly): installs the real DashAI (the latest release and the
  oldest supported minor) and runs `tests/compat`.

## Installing DashAI

DashAI needs **Python 3.11 or newer**. These are the ways to get it; the official
documentation lives at <https://github.com/DashAISoftware/DashAI> and <https://dash-ai.com/>.

### With uv (recommended)

[uv](https://docs.astral.sh/uv/getting-started/installation/) installs Python for you and
keeps everything isolated.

```bash
# DashAI as an app, with this plugin, in its own environment
uv tool install dashai --with dashai-ollama
dashai
```

Or inside a project environment:

```bash
uv venv --python 3.12
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install dashai dashai-ollama
dashai
```

### With pip and PyPI

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install dashai dashai-ollama
dashai
```

### Desktop installers and Docker

DashAI also ships desktop installers (Windows, macOS, Linux AppImage) and Docker images; see
its [releases](https://github.com/DashAISoftware/DashAI/releases/latest). With those you
cannot `pip install` next to DashAI, so add the plugin from its interface instead:

1. Open the **Plugins** module.
2. Search for `dashai-ollama` and install it.

DashAI installs plugins in `~/.DashAI/plugins/` (override with `DASHAI_LOCAL_PATH`), so they
survive upgrades and work the same way in every installation method.

### GPU builds

The default install runs on CPU. For NVIDIA (CUDA) or AMD (ROCm) acceleration of DashAI's
own models, follow the PyTorch steps in the DashAI documentation. **That does not affect this
plugin**: the GPU used for Ollama models is whichever one Ollama itself uses.

### Running DashAI

```bash
dashai
```

This starts the server and opens the graphical interface at <http://localhost:8000>. Use
`dashai --help` for the options (for example `--no-browser` or `--logging-level DEBUG`).

## Using the plugin in DashAI

### Chat

1. Open **Generative** and start a new session.
2. Pick **Ollama Model** in the model gallery. It needs no download from DashAI: Ollama holds the
   weights.
3. Fill in the form (see [Parameters](#parameters)). Hover over the **?** next to each field
   for a short explanation.
4. Create the session and start chatting. Each message is sent with the conversation history.

### RAG

A RAG pipeline has two places where this plugin fits:

- **Generation model:** choose **Ollama Model**. This is the model that writes the answer.
- **Embedding model:** in a dense retriever, choose **Ollama Embedding** and use an
  embedding model such as `nomic-embed-text`. Chat models do not work as embeddings.

RAG puts the retrieved document chunks in every prompt, so give the **generation** model a
context window that fits them (16,384 tokens, the default, is a good start).

> **Re-index after changing the embedding model.** An index is tied to the model, and the
> vector size, that built it. If you change `model` in **Ollama Embedding**, create the
> index again.

### Using it from another machine

Ollama listens on `localhost` only. To use a server on another machine, start it with
`OLLAMA_HOST=0.0.0.0` and put its address in **Server URL**, for example
`http://192.168.1.20:11434`. Ollama has no authentication, so only do this on a network you
trust.

## Parameters

The **Ollama Model** component (`OllamaModel`):

| Parameter | Default | What it does |
| --- | --- | --- |
| **Server URL** (`host`) | `http://localhost:11434` | Address of the Ollama server, with protocol and port, no trailing slash. Change it only if Ollama runs on another machine or port. |
| **Model** (`model`) | `bonsai-2:9b` | Exact name of the model, **including its tag**, as `ollama list` shows it (for example `qwen3:8b`). It must be pulled first. |
| **Max tokens** (`max_tokens`) | `4096` | Maximum new tokens per reply (Ollama's `num_predict`). About 1 token is 0.75 English words. Use 256-1,024 for short answers and 2,048-4,096 for detailed ones or code. **Reasoning models also spend tokens thinking, so give them 4,096 or more.** |
| **Temperature** (`temperature`) | `0.7` | Randomness, from 0.0 to 2.0. Use 0.0-0.3 for facts, extraction and RAG, 0.6-0.8 for chat, 1.0 or more for creative writing. |
| **Top P** (`top_p`) | `0.9` | Nucleus sampling, from 0.0 to 1.0: only the most likely tokens whose probabilities add up to this value are considered. Change this *or* the temperature, not both. |
| **Top K** (`top_k`) | `40` | Only the K most likely tokens are considered at each step. `0` disables it. |
| **Context window** (`context_window`) | `16384` | Total tokens the model keeps in memory (`num_ctx`): history, retrieved documents and the reply. `8192` for short chats on low-memory computers, `16384` for chat and RAG on most computers, `32768` or more for long documents. Memory grows with this value, and what exceeds it is silently dropped. |
| **Thinking** (`think`) | `auto` | Whether the model reasons step by step before answering. `auto` leaves it to the model, `on` and `off` force it. Only for models with the `thinking` capability: on others, `on` and `off` fail, so keep `auto`. |
| **Keep alive** (`keep_alive`) | `5m` | How long Ollama keeps the model in memory after the last request. A loaded model answers faster; an unloaded one frees memory. `30m` or `1h` for a work session, `-1` to never unload, `0` to unload right after each reply. |
| **Timeout** (`timeout`) | `600` | Seconds to wait for the server before giving up. The first request after a model is loaded is slower. Replies are not streamed, so large or reasoning models on CPU may need more. |

The **Ollama Embedding** component (`OllamaEmbedding`):

| Parameter | Default | What it does |
| --- | --- | --- |
| **Server URL** (`host`) | `http://localhost:11434` | Same as above. |
| **Model** (`model`) | `nomic-embed-text` | Name of an **embedding** model, as `ollama list` shows it. Chat models do not work here. |
| **Batch size** (`batch_size`) | `32` | How many text chunks are sent per request while indexing. `16`-`64` suits most computers; lower it if Ollama runs out of memory. |
| **Timeout** (`timeout`) | `120` | Seconds to wait for the server on each request. |

## Where to find the values

| You need | Where to look |
| --- | --- |
| The exact **model name** | `ollama list`, or the [library](https://ollama.com/library) (the tag list is at `https://ollama.com/library/<model>/tags`). |
| Which **sizes** a model comes in, and its notes | Its page in the [library](https://ollama.com/library), for example <https://ollama.com/library/qwen3>. Model pages usually recommend `temperature`, `top_p` and `top_k`. |
| The model's **maximum context length** | `ollama show <model>`, under *context length*. Do not set **Context window** above it. |
| Whether it can **think**, use tools, see images or embed | `ollama show <model>`, under *Capabilities* (`thinking` and `embedding` are the ones this plugin uses). The library page shows the same as badges. |
| Which **thinking levels** it accepts | `ollama show <model>`, under *thinking > levels*. |
| What is **loaded in memory** and for how long | `ollama ps`. |
| Whether the **server is up** and its version | `curl http://localhost:11434/api/version`, or `ollama --version`. |
| The **embedding models** | The library filtered by embedding: <https://ollama.com/search?c=embedding>. |

## Recommended models

A starting list taken from the [Ollama library](https://ollama.com/library) on
**2026-11-03**: capabilities, parameter sizes and downloads (*pulls*) come from the catalog,
and the *good for* column is a judgement, not an Ollama claim. The catalog changes quickly,
so check the library for newer releases.

As a rough rule of thumb (not from Ollama), a model in the default 4-bit build needs about
0.6 GB of RAM or VRAM per billion parameters, plus the memory of the context window. The size
that `ollama list` prints after a pull is the real figure for your download.

### Chat and RAG generation

| Model | Sizes | Capabilities | Pulls | Good for |
| --- | --- | --- | --- | --- |
| `bonsai-2` | 2b, 4b, 9b, 27b | vision, tools, thinking | 104.6K | Small footprint with reasoning; used while developing this plugin. |
| `llama3.2` | 1b, 3b | tools | 86.9M | A fast first model on a laptop or CPU. |
| `qwen3` | 0.6b to 235b | tools, thinking | 42.4M | General use at almost any size, with optional reasoning. |
| `qwen3.5` | 0.8b to 122b | vision, tools, thinking | 24.3M | A newer all-rounder across many sizes. |
| `gemma3` | 270m to 27b | vision | 41.6M | A strong model that runs on a single GPU. |
| `gemma4` | e2b, e4b, 12b, 26b, 31b | vision, tools, thinking, audio | 28.5M | Reasoning and coding at each size. |
| `smollm2` | 135m, 360m, 1.7b | tools | 4.2M | Very small machines, tests and quick experiments. |
| `mistral-nemo` | 12b | tools | 5.8M | A 12B model with a 128k context. |
| `phi4` | 14b | | 7.8M | A compact 14B model from Microsoft. |

### Reasoning

| Model | Sizes | Capabilities | Pulls | Good for |
| --- | --- | --- | --- | --- |
| `deepseek-r1` | 1.5b to 671b | tools, thinking | 94.7M | Step-by-step reasoning problems. |
| `gpt-oss` | 20b, 120b | tools, thinking | 14.6M | Strong reasoning when you have the memory for it. |
| `qwen3` | 0.6b to 235b | tools, thinking | 42.4M | Reasoning that you can switch off with **Thinking**. |

Reasoning models need a large **Max tokens** (4,096 or more) or **Thinking** set to *Off*.

### Code

| Model | Sizes | Capabilities | Pulls |
| --- | --- | --- | --- |
| `qwen2.5-coder` | 0.5b to 32b | tools | 23.3M |
| `qwen3-coder` | 30b, 480b | tools | 10.6M |

### Embeddings (for RAG)

| Model | Sizes | Pulls | Notes |
| --- | --- | --- | --- |
| `nomic-embed-text` | | 95.5M | The default here: a high-performing model with a large token context window. |
| `mxbai-embed-large` | 335m | 16.5M | A larger model from mixedbread.ai. |
| `bge-m3` | 567m | 8.2M | From BAAI, described as multi-lingual and multi-granularity; a good pick for non-English documents. |
| `qwen3-embedding` | 0.6b, 4b, 8b | 4.5M | A family of sizes from the Qwen3 series. |
| `embeddinggemma` | 300m | 2.7M | A 300M-parameter model from Google. |
| `all-minilm` | 22m, 33m | 3.9M | Tiny and fast, for small machines. |

Models with the `vision` capability can read images, but DashAI's text chat does not send
images yet, so that capability is not used by this plugin.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| *Failed to connect to Ollama* | The server is not running or **Server URL** is wrong. Run `ollama serve` (or start the Ollama app) and check `curl http://localhost:11434/api/version`. |
| *model '...' not found (404)* | The model is not downloaded, or the name is wrong. Run `ollama pull <model>` and copy the exact name, with its tag, from `ollama list`. |
| *hit max_tokens before producing an answer* | A reasoning model used all its tokens thinking. Raise **Max tokens** (4,096 or more) or set **Thinking** to *Off*. |
| An error when **Thinking** is `on` or `off` | The model has no `thinking` capability. Set **Thinking** back to `auto`. |
| The first reply is very slow | Ollama is loading the model into memory. Later replies are faster; raise **Keep alive** to keep it loaded. |
| A request times out | Raise **Timeout**, or use a smaller model. Replies are not streamed, so the whole answer must finish inside the timeout. |
| RAG answers ignore the documents | The context window may be too small for the retrieved chunks, so raise **Context window**; or the index was built with a different embedding model, so re-index. |
| *requires newer version of Ollama* | Update Ollama (`ollama --version`); some models need a recent release. |

## Development

To change the plugin, work on a branch and open a pull request against `main` (see
[Conventions](#commits-and-pull-requests)). You need
[uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.11 or newer (uv
installs it for you).

```bash
git clone https://github.com/idia-tt/dashai-ollama.git
cd dashai-ollama
uv sync                      # creates .venv with the runtime and dev dependencies
uv run pre-commit install    # installs the git hooks (pre-commit, commit-msg and pre-push)
```

| Command | What it does |
| --- | --- |
| `uv run pytest` | Runs the tests, with coverage |
| `uv run ruff check .` | Lints (add `--fix` to apply safe fixes) |
| `uv run ruff format .` | Formats |
| `uv run ty check` | Type checks with [ty](https://docs.astral.sh/ty/) |
| `uv run pre-commit run --all-files` | Runs every hook on every file |
| `uv run cz commit` | Writes a conventional commit through prompts |
| `uv build` | Builds the wheel and the sdist into `dist/` |

### Layout

```text
src/dashai_ollama/
    ollama_model.py        OllamaModel: chat and RAG generation
    ollama_embedding.py    OllamaEmbedding: dense embeddings for RAG
stubs/DashAI/              a minimal stand-in for the DashAI host (see below)
tests/                     unit tests, no Ollama or DashAI needed
.github/workflows/ci.yml   lint, type check, tests, build and release
pyproject.toml             package, entry points, ruff, ty, pytest and commitizen
```

### How the plugin is discovered

`pyproject.toml` declares one entry point per component in the `dashai.plugins` group.
DashAI reads that group at start up and registers each class:

```toml
[project.entry-points.'dashai.plugins']
OllamaModel = 'dashai_ollama.ollama_model:OllamaModel'
OllamaEmbedding = 'dashai_ollama.ollama_embedding:OllamaEmbedding'
```

### Tests run against stubs

DashAI is the host application, not a dependency, so it is not installed in development or in
CI. `stubs/DashAI/` is a small stand-in for the few DashAI names the plugin imports. pytest
and ty both put it on their path. See [`stubs/README.md`](stubs/README.md). Add to it when the
plugin starts using a new DashAI name.

Because the stubs are not DashAI, `tests/compat` has a few tests that run against the real
host (registration, the schemas, the embedding being listed as a dense embedding). They are
deselected by default and run by the weekly compatibility workflow. To run them yourself,
install the plugin editable into the environment that runs DashAI and run:

```bash
uv pip install -e /path/to/dashai-ollama pytest   # inside DashAI's environment
pytest -m dashai -o pythonpath=src -o addopts=""   # from the plugin's directory
```

The form in the interface and a full RAG run are still checked by hand. To try the plugin in
a real DashAI, install it editable the same way, then start DashAI:

```bash
uv pip install -e /path/to/dashai-ollama   # inside DashAI's environment
dashai
```

### Adding a parameter

1. Add the field to the schema in `ollama_model.py` or `ollama_embedding.py`, with a default
   as a module constant.
2. Write its `description` and `alias` in **all five languages** (`en`, `es`, `pt`, `de`,
   `zh`). Parameter descriptions are shown in a tooltip that renders Markdown (bold, inline
   code, lists and `[text](url)` links); put blank lines before lists, and write links as
   Markdown because bare URLs are not clickable.
3. Use the value in the request, and add a test.
4. Update the tables in this README.

The component `DESCRIPTION` (shown in the model gallery) is plain text: Markdown is not
rendered there.

## Conventions

### Commits and pull requests

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<optional scope>): <subject in the imperative mood>
```

The commit-msg hook, CI and the release all rely on it, so a message that does not match is
rejected. `uv run cz commit` asks for each part. Typical scopes are `model`, `embedding`
and `deps`.

| Type | Use it for | Release |
| --- | --- | --- |
| `feat` | A new feature or parameter | **minor** |
| `perf` | A performance improvement | **minor** |
| `fix` | A bug fix | patch |
| `refactor` | A code change that is neither a fix nor a feature | patch |
| `style` | Formatting or lint changes, no behavior change | patch |
| `chore` | Maintenance of the code or its dependencies | patch |
| `build` | Packaging or build configuration | patch |
| `docs` | Documentation only | none |
| `test` | Tests only | none |
| `ci` | CI configuration only | none |
| `revert` | Reverting an earlier commit | none |

A breaking change bumps the **major** version. Mark it with `!` after the type
(`feat(model)!: rename the host field`) or with a `BREAKING CHANGE:` footer.

**Every change to the plugin goes through a pull request against `main`**: create a branch,
push it and open a PR. Do not push to `main` directly, because the release job pushes the
version bump there. CI checks every commit message and the **PR title**, so write the title
as a conventional commit too. It becomes the commit message if the PR is squash merged. Name
branches after the type, for example `feat/keep-alive` or `fix/timeout`.

### Code

- [ruff](https://docs.astral.sh/ruff/) lints and formats with a small core rule set (see
  `pyproject.toml`); [ty](https://docs.astral.sh/ty/) type checks. Both run in the hooks and
  in CI.
- Use NumPy style docstrings, as DashAI does.
- Every user-facing string is translated into `en`, `es`, `pt`, `de` and `zh`.
- Defaults are module constants shared by the schema placeholder and the code.
- Errors that reach the user are `RuntimeError`s with a message that says what to do.

## Releases

Releases are automatic. When a pull request is merged into `main`, CI runs the checks and
then the release job:

1. Reads the conventional commits since the last tag and decides the next
   [semantic version](https://semver.org/) from the table above. If there are only `docs`,
   `test` or `ci` commits, nothing is released.
2. Updates the version in `pyproject.toml` and `uv.lock` and the `CHANGELOG.md`, commits it
   as `bump: version X → Y` and creates the tag `vY`.
3. Builds the wheel and sdist and creates a **GitHub Release** with them attached and the
   changelog entry as notes.

On a pull request, CI also posts a preview of the release the PR would produce.

While the version is `0.x`, a breaking change still bumps to `1.0.0`: this project follows
the rule strictly.

## License

[MIT](LICENSE).
