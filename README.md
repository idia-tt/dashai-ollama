# dashai-ollama

DashAI plugin that runs chat and RAG against a local [Ollama](https://ollama.com) server.

## Use

1. Start Ollama and pull a model: `ollama serve`, then `ollama pull llama3.2:3b`.
2. In DashAI, open the **Plugins** module and install `dashai-ollama`
   (or `pip install dashai-ollama` next to DashAI).
3. Create a generative session and pick **Ollama**. Set the server URL (default
   `http://localhost:11434`) and the model name exactly as `ollama list` shows it.

Reasoning models spend `max_tokens` on hidden thinking first. If you get a
"hit max_tokens" error, raise it.

## Limits

No streaming: the full reply arrives at once, so large models on CPU can be slow.
