"""Text generation through a local Ollama server."""

from typing import List

import httpx
import ollama

from DashAI.back.core.schema_fields import (
    BaseSchema,
    float_field,
    int_field,
    schema_field,
    string_field,
)
from DashAI.back.core.utils import MultilingualString
from DashAI.back.models.text_to_text_generation_model import (
    TextToTextGenerationTaskModel,
)

DEFAULT_HOST = "http://localhost:11434"


class OllamaSchema(BaseSchema):
    """Hyperparameters of a model served by a local Ollama instance."""

    host: schema_field(
        string_field(),
        placeholder=DEFAULT_HOST,
        description=MultilingualString(
            en="URL of the Ollama server. The default is a local instance.",
            es="URL del servidor de Ollama. Por defecto es una instancia local.",
            pt="URL do servidor Ollama. O padrão é uma instância local.",
            de="URL des Ollama-Servers. Standard ist eine lokale Instanz.",
            zh="Ollama 服务器的 URL。默认为本地实例。",
        ),
        alias=MultilingualString(
            en="Server URL",
            es="URL del servidor",
            pt="URL do servidor",
            de="Server-URL",
            zh="服务器 URL",
        ),
    )  # type: ignore

    model: schema_field(
        string_field(),
        placeholder="llama3.2:3b",
        description=MultilingualString(
            en=(
                "Name of the model as shown by `ollama list`. It must be pulled "
                "beforehand with `ollama pull <model>`."
            ),
            es=(
                "Nombre del modelo tal como aparece en `ollama list`. Debe "
                "haberse descargado antes con `ollama pull <modelo>`."
            ),
            pt=(
                "Nome do modelo como mostrado por `ollama list`. Deve ser "
                "baixado antes com `ollama pull <modelo>`."
            ),
            de=(
                "Name des Modells wie in `ollama list`. Es muss vorher mit "
                "`ollama pull <Modell>` heruntergeladen werden."
            ),
            zh="`ollama list` 中显示的模型名称。必须先用 `ollama pull <模型>` 下载。",
        ),
        alias=MultilingualString(
            en="Model",
            es="Modelo",
            pt="Modelo",
            de="Modell",
            zh="模型",
        ),
    )  # type: ignore

    max_tokens: schema_field(
        int_field(ge=1),
        placeholder=512,
        description=MultilingualString(
            en="Maximum number of new tokens generated per response.",
            es="Número máximo de tokens nuevos generados por respuesta.",
            pt="Número máximo de tokens novos gerados por resposta.",
            de="Maximale Anzahl neuer Token pro Antwort.",
            zh="每次响应生成的最大新 token 数量。",
        ),
        alias=MultilingualString(
            en="Max tokens",
            es="Tokens máximos",
            pt="Tokens máximos",
            de="Maximale neue Token",
            zh="最大 token 数",
        ),
    )  # type: ignore

    temperature: schema_field(
        float_field(ge=0.0, le=2.0),
        placeholder=0.7,
        description=MultilingualString(
            en="Sampling temperature. 0.0 is deterministic; higher is more varied.",
            es=(
                "Temperatura de muestreo. 0.0 es determinista; valores más altos "
                "dan salidas más variadas."
            ),
            pt=(
                "Temperatura de amostragem. 0.0 é determinístico; valores "
                "maiores dão saídas mais variadas."
            ),
            de=(
                "Stichprobentemperatur. 0.0 ist deterministisch; höhere Werte "
                "sind abwechslungsreicher."
            ),
            zh="采样温度。0.0 为确定性输出；值越高输出越多样。",
        ),
        alias=MultilingualString(
            en="Temperature",
            es="Temperatura",
            pt="Temperatura",
            de="Temperatur",
            zh="温度",
        ),
    )  # type: ignore

    context_window: schema_field(
        int_field(ge=1),
        placeholder=4096,
        description=MultilingualString(
            en=(
                "Context window in tokens (`num_ctx`), covering prompt and "
                "response. RAG needs a larger value than short chats; it uses "
                "more memory."
            ),
            es=(
                "Ventana de contexto en tokens (`num_ctx`), que incluye prompt y "
                "respuesta. RAG necesita más que un chat corto; consume más "
                "memoria."
            ),
            pt=(
                "Janela de contexto em tokens (`num_ctx`), incluindo prompt e "
                "resposta. RAG precisa de mais que um chat curto; usa mais "
                "memória."
            ),
            de=(
                "Kontextfenster in Token (`num_ctx`), inklusive Prompt und "
                "Antwort. RAG braucht mehr als kurze Chats; benötigt mehr Speicher."
            ),
            zh=(
                "上下文窗口（`num_ctx`），包含提示词和响应。"
                "RAG 需要比短对话更大的值，占用更多内存。"
            ),
        ),
        alias=MultilingualString(
            en="Context window",
            es="Ventana de contexto",
            pt="Janela de contexto",
            de="Kontextfenster",
            zh="上下文窗口",
        ),
    )  # type: ignore

    timeout: schema_field(
        int_field(ge=1),
        placeholder=300,
        description=MultilingualString(
            en=(
                "Seconds to wait for the server before failing. Large models on "
                "CPU can be slow, and there is no streaming."
            ),
            es=(
                "Segundos de espera al servidor antes de fallar. Los modelos "
                "grandes en CPU pueden ser lentos, y no hay streaming."
            ),
            pt=(
                "Segundos de espera pelo servidor antes de falhar. Modelos "
                "grandes em CPU podem ser lentos, e não há streaming."
            ),
            de=(
                "Sekunden Wartezeit auf den Server vor dem Fehler. Große Modelle "
                "auf der CPU können langsam sein, und es gibt kein Streaming."
            ),
            zh=(
                "等待服务器响应的秒数，超时则失败。"
                "CPU 上的大模型可能很慢，且不支持流式输出。"
            ),
        ),
        alias=MultilingualString(
            en="Timeout (s)",
            es="Tiempo máximo (s)",
            pt="Tempo limite (s)",
            de="Zeitlimit (s)",
            zh="超时（秒）",
        ),
    )  # type: ignore


class OllamaModel(TextToTextGenerationTaskModel):
    """Chat model served by a local Ollama instance.

    The weights live in Ollama, not in DashAI, so this component has no
    download step: pull the model with ``ollama pull`` first. Requires the
    ``dashai-ollama`` plugin (``pip install dashai-ollama``).
    """

    SCHEMA = OllamaSchema
    COLOR: str = "#455a64"
    DISPLAY_NAME = MultilingualString(
        en="Ollama",
        es="Ollama",
        pt="Ollama",
        de="Ollama",
        zh="Ollama",
    )
    DESCRIPTION = MultilingualString(
        en=(
            "Runs any model served by a local Ollama server. Start it with "
            "`ollama serve` and pull a model with `ollama pull <model>`."
        ),
        es=(
            "Usa cualquier modelo servido por un servidor Ollama local. Inícielo "
            "con `ollama serve` y descargue un modelo con `ollama pull "
            "<modelo>`."
        ),
        pt=(
            "Usa qualquer modelo servido por um servidor Ollama local. Inicie-o "
            "com `ollama serve` e baixe um modelo com `ollama pull <modelo>`."
        ),
        de=(
            "Nutzt jedes Modell eines lokalen Ollama-Servers. Starten mit "
            "`ollama serve`, Modell laden mit `ollama pull <Modell>`."
        ),
        zh=(
            "使用本地 Ollama 服务器提供的任意模型。用 `ollama serve` 启动，"
            "用 `ollama pull <模型>` 下载模型。"
        ),
    )

    def __init__(self, **kwargs):
        """Create the client for the configured Ollama server.

        """
        kwargs = self.validate_and_transform(kwargs)
        self.host = kwargs.get("host", DEFAULT_HOST)
        self.model_name = kwargs["model"]
        self.options = {
            "num_predict": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "num_ctx": kwargs.get("context_window", 4096),
        }
        self._error_types = (ollama.ResponseError, ConnectionError, httpx.HTTPError)
        self.client = ollama.Client(host=self.host, timeout=kwargs.get("timeout", 300))

    def generate(self, prompt: list) -> List[str]:
        """Generate a reply for a conversation.

        Parameters
        ----------
        prompt : list of dict
            Conversation in chat format, each dict with ``role`` and
            ``content``.

        Returns
        -------
        list of str
            A single-element list with the model's reply.

        Raises
        ------
        RuntimeError
            If the server is unreachable or rejects the request (for example,
            because the model was not pulled).
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=prompt,
                options=self.options,
            )
        except self._error_types as e:
            raise RuntimeError(
                f"Ollama request to {self.host} failed for model "
                f"'{self.model_name}': {e}. Is the server running and the model "
                "pulled?"
            ) from e
        content = response["message"]["content"]
        if not content and response.get("done_reason") == "length":
            # Reasoning models spend max_tokens on hidden thinking first.
            raise RuntimeError(
                f"Model '{self.model_name}' hit max_tokens "
                f"({self.options['num_predict']}) before producing an answer. "
                "Reasoning models need a larger max_tokens."
            )
        return [content]
