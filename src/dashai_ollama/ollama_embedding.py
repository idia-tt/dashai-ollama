"""Dense embeddings computed by a local Ollama server, for DashAI RAG."""

from typing import List

import httpx
import numpy as np
import ollama

from DashAI.back.core.schema_fields import (
    BaseSchema,
    int_field,
    schema_field,
    string_field,
)
from DashAI.back.core.utils import MultilingualString
from DashAI.back.models.RAG.embeddings.dense_embedding import DenseEmbedding

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_BATCH_SIZE = 32
DEFAULT_TIMEOUT = 120


class OllamaEmbeddingSchema(BaseSchema):
    """Parameters of an embedding model served by a local Ollama instance."""

    host: schema_field(
        string_field(),
        placeholder=DEFAULT_HOST,
        description=MultilingualString(
            en=(
                "Address of the Ollama server, including protocol and port, "
                "without a trailing slash. Change it only if the server runs "
                "on another machine or port."
            ),
            es=(
                "Dirección del servidor Ollama, con protocolo y puerto, sin "
                "barra final. Cámbiela solo si el servidor corre en otra "
                "máquina o puerto."
            ),
            pt=(
                "Endereço do servidor Ollama, com protocolo e porta, sem barra "
                "final. Altere apenas se o servidor rodar em outra máquina ou "
                "porta."
            ),
            de=(
                "Adresse des Ollama-Servers mit Protokoll und Port, ohne "
                "abschließenden Schrägstrich. Nur ändern, wenn der Server auf "
                "einem anderen Rechner oder Port läuft."
            ),
            zh=(
                "Ollama 服务器的地址，包含协议和端口，末尾不带斜杠。"
                "仅当服务器运行在其他机器或端口时才需要修改。"
            ),
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
        placeholder=DEFAULT_MODEL,
        description=MultilingualString(
            en=(
                "Name of an **embedding** model, as shown by `ollama list` "
                "(for example `nomic-embed-text`). Chat models do not work "
                "here. Download it first with `ollama pull <model>`. Browse "
                "the [Ollama library](https://ollama.com/search?c=embedding)."
                " An index is tied to the model that built it: if you change "
                "the model, build the index again."
            ),
            es=(
                "Nombre de un modelo de **embeddings**, tal como aparece en "
                "`ollama list` (por ejemplo `nomic-embed-text`). Los modelos "
                "de chat no sirven aquí. Descárguelo antes con `ollama pull "
                "<modelo>`. Explore la [biblioteca de Ollama]"
                "(https://ollama.com/search?c=embedding). Un índice queda "
                "ligado al modelo que lo creó: si cambia el modelo, vuelva a "
                "crear el índice."
            ),
            pt=(
                "Nome de um modelo de **embeddings**, como mostrado por "
                "`ollama list` (por exemplo `nomic-embed-text`). Modelos de "
                "chat não servem aqui. Baixe-o antes com `ollama pull "
                "<modelo>`. Veja a [biblioteca do Ollama]"
                "(https://ollama.com/search?c=embedding). Um índice fica "
                "preso ao modelo que o criou: se trocar o modelo, crie o "
                "índice novamente."
            ),
            de=(
                "Name eines **Embedding**-Modells, wie in `ollama list` "
                "angezeigt (z. B. `nomic-embed-text`). Chat-Modelle "
                "funktionieren hier nicht. Zuvor mit `ollama pull <Modell>` "
                "laden. Siehe die [Ollama-Bibliothek]"
                "(https://ollama.com/search?c=embedding). Ein Index ist an das "
                "Modell gebunden, das ihn erstellt hat: Bei einem Wechsel den "
                "Index neu erstellen."
            ),
            zh=(
                "**嵌入**模型的名称，与 `ollama list` 中显示的一致"
                "（例如 `nomic-embed-text`）。聊天模型在此无效。"
                "请先用 `ollama pull <模型>` 下载。浏览 [Ollama 模型库]"
                "(https://ollama.com/search?c=embedding)。"
                "索引与创建它的模型绑定：更换模型后需重新创建索引。"
            ),
        ),
        alias=MultilingualString(
            en="Model",
            es="Modelo",
            pt="Modelo",
            de="Modell",
            zh="模型",
        ),
    )  # type: ignore

    batch_size: schema_field(
        int_field(ge=1),
        placeholder=DEFAULT_BATCH_SIZE,
        description=MultilingualString(
            en=(
                "How many text chunks are sent to Ollama in each request "
                "while indexing documents.\n\n"
                "- **16-64**: good for most computers\n"
                "- **Lower**: if Ollama runs out of memory\n"
                "- **Higher**: faster indexing on a powerful GPU"
            ),
            es=(
                "Cuántos fragmentos de texto se envían a Ollama en cada "
                "solicitud al indexar documentos.\n\n"
                "- **16-64**: adecuado para la mayoría de los computadores\n"
                "- **Menos**: si Ollama se queda sin memoria\n"
                "- **Más**: indexación más rápida con una GPU potente"
            ),
            pt=(
                "Quantos trechos de texto são enviados ao Ollama em cada "
                "solicitação ao indexar documentos.\n\n"
                "- **16-64**: adequado para a maioria dos computadores\n"
                "- **Menos**: se o Ollama ficar sem memória\n"
                "- **Mais**: indexação mais rápida com uma GPU potente"
            ),
            de=(
                "Wie viele Textabschnitte beim Indexieren von Dokumenten pro "
                "Anfrage an Ollama gesendet werden.\n\n"
                "- **16-64**: für die meisten Rechner geeignet\n"
                "- **Weniger**: wenn Ollama der Speicher ausgeht\n"
                "- **Mehr**: schnelleres Indexieren mit leistungsstarker GPU"
            ),
            zh=(
                "索引文档时，每次请求发送给 Ollama 的文本片段数量。\n\n"
                "- **16-64**：适合大多数电脑\n"
                "- **更小**：Ollama 内存不足时\n"
                "- **更大**：在高性能 GPU 上索引更快"
            ),
        ),
        alias=MultilingualString(
            en="Batch size",
            es="Tamaño de lote",
            pt="Tamanho do lote",
            de="Batch-Größe",
            zh="批大小",
        ),
    )  # type: ignore

    timeout: schema_field(
        int_field(ge=1),
        placeholder=DEFAULT_TIMEOUT,
        description=MultilingualString(
            en=(
                "Seconds to wait for the server on each request before giving "
                "up. The first request after the model is downloaded or "
                "unloaded is slower because Ollama must load it."
            ),
            es=(
                "Segundos de espera al servidor en cada solicitud antes de "
                "rendirse. La primera solicitud tras descargar o descargar de "
                "memoria el modelo es más lenta porque Ollama debe cargarlo."
            ),
            pt=(
                "Segundos de espera pelo servidor em cada solicitação antes de "
                "desistir. A primeira solicitação após baixar ou descarregar o "
                "modelo é mais lenta porque o Ollama precisa carregá-lo."
            ),
            de=(
                "Sekunden Wartezeit auf den Server pro Anfrage, bevor "
                "abgebrochen wird. Die erste Anfrage nach dem Laden oder "
                "Entladen des Modells ist langsamer, da Ollama es laden muss."
            ),
            zh=(
                "每次请求等待服务器的秒数，超时则放弃。模型刚下载或被卸载后，"
                "第一次请求会更慢，因为 Ollama 需要加载它。"
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


class OllamaEmbedding(DenseEmbedding):
    """Dense embeddings computed by a local Ollama server.

    The weights live in Ollama, so there is no download step here: pull an
    embedding model with ``ollama pull`` first.
    """

    SCHEMA = OllamaEmbeddingSchema
    DISPLAY_NAME: str = MultilingualString(
        en="Ollama Embedding",
        es="Embedding Ollama",
        pt="Embedding Ollama",
        de="Ollama-Embedding",
        zh="Ollama 嵌入",
    )
    DESCRIPTION: str = MultilingualString(
        en=(
            "Dense embeddings computed by a local Ollama server. Pull an "
            "embedding model with `ollama pull nomic-embed-text` and use its "
            "name in the Model field."
        ),
        es=(
            "Embeddings densos calculados por un servidor Ollama local. "
            "Descargue un modelo de embeddings con `ollama pull "
            "nomic-embed-text` y use su nombre en el campo Modelo."
        ),
        pt=(
            "Embeddings densos calculados por um servidor Ollama local. Baixe "
            "um modelo de embeddings com `ollama pull nomic-embed-text` e use "
            "o nome dele no campo Modelo."
        ),
        de=(
            "Dichte Embeddings, berechnet von einem lokalen Ollama-Server. "
            "Laden Sie ein Embedding-Modell mit `ollama pull nomic-embed-text` "
            "und tragen Sie seinen Namen im Feld Modell ein."
        ),
        zh=(
            "由本地 Ollama 服务器计算的稠密嵌入。用 "
            "`ollama pull nomic-embed-text` 下载嵌入模型，并在“模型”字段中填入其名称。"
        ),
    )

    def __init__(self, **kwargs):
        """Create the client for the configured Ollama server."""
        self.params = self.validate_and_transform(kwargs)
        self.host = self.params.get("host", DEFAULT_HOST)
        self.model_name = self.params["model"]
        self.batch_size = self.params.get("batch_size", DEFAULT_BATCH_SIZE)
        self._error_types = (ollama.ResponseError, ConnectionError, httpx.HTTPError)
        self.client = ollama.Client(
            host=self.host, timeout=self.params.get("timeout", DEFAULT_TIMEOUT)
        )

    def load(self):
        """Check the server and model, and learn the embedding dimension.

        Raises
        ------
        RuntimeError
            If the server is unreachable or the model is not available.
        """
        self.embedding_dim = self._embed(["dimension probe"]).shape[1]

    def encode(self, text: str) -> np.ndarray:
        """Embed one text; returns a float32 vector of shape ``(dim,)``."""
        return self._embed([text])[0]

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """Embed many texts; returns a float32 array of shape ``(n, dim)``."""
        batches = [
            self._embed(texts[i : i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.concatenate(batches) if batches else np.empty((0, 0), np.float32)

    def save(self):
        """No-op. Persistence is handled externally."""

    def train(self, **kwargs):
        """No-op. Embeddings are computed by a pretrained model."""

    def _embed(self, texts: List[str]) -> np.ndarray:
        try:
            response = self.client.embed(model=self.model_name, input=texts)
        except self._error_types as e:
            raise RuntimeError(
                f"Ollama embedding request to {self.host} failed for model "
                f"'{self.model_name}': {e}. Is the server running and an "
                "embedding model pulled?"
            ) from e
        vectors = np.asarray(response["embeddings"], dtype=np.float32)
        self.embedding_dim = vectors.shape[1]
        return vectors
