"""Text generation through a local Ollama server."""

import httpx
import ollama
from DashAI.back.core.schema_fields import (
    BaseSchema,
    enum_field,
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
DEFAULT_MODEL = "bonsai-2:9b"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_CONTEXT_WINDOW = 16384
DEFAULT_THINK = "auto"
DEFAULT_KEEP_ALIVE = "5m"
DEFAULT_TIMEOUT = 600


def _parse_keep_alive(value: str) -> "str | float":
    """Turn ``"-1"`` or ``"0"`` into numbers: Ollama rejects them as durations."""
    try:
        return float(value)
    except ValueError:
        return value


class OllamaSchema(BaseSchema):
    """Hyperparameters of a model served by a local Ollama instance."""

    host: schema_field(
        string_field(),
        placeholder=DEFAULT_HOST,
        description=MultilingualString(
            en=(
                "Address of the Ollama server, including protocol and port, "
                "without a trailing slash. Ollama listens on "
                "http://localhost:11434 by default. Change it only if the "
                "server runs on another machine or port."
            ),
            es=(
                "Dirección del servidor Ollama, con protocolo y puerto, sin "
                "barra final. Ollama escucha por defecto en "
                "http://localhost:11434. Cámbiela solo si el servidor corre "
                "en otra máquina o puerto."
            ),
            pt=(
                "Endereço do servidor Ollama, com protocolo e porta, sem barra "
                "final. O Ollama escuta por padrão em http://localhost:11434. "
                "Altere apenas se o servidor rodar em outra máquina ou porta."
            ),
            de=(
                "Adresse des Ollama-Servers mit Protokoll und Port, ohne "
                "abschließenden Schrägstrich. Ollama lauscht standardmäßig auf "
                "http://localhost:11434. Nur ändern, wenn der Server auf einem "
                "anderen Rechner oder Port läuft."
            ),
            zh=(
                "Ollama 服务器的地址，包含协议和端口，末尾不带斜杠。"
                "Ollama 默认监听 http://localhost:11434。"
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
    )  # ty: ignore[invalid-type-form]

    model: schema_field(
        string_field(),
        placeholder=DEFAULT_MODEL,
        description=MultilingualString(
            en=(
                "Exact name of the model, including its tag, as shown by "
                "`ollama list` (for example `llama3.2:3b` or `qwen3:8b`). "
                "It must be downloaded first with `ollama pull <model>`. "
                "Browse the [Ollama library](https://ollama.com/library)."
            ),
            es=(
                "Nombre exacto del modelo, con su etiqueta, tal como aparece "
                "en `ollama list` (por ejemplo `llama3.2:3b` o `qwen3:8b`). "
                "Debe descargarse antes con `ollama pull <modelo>`. "
                "Explore la [biblioteca de Ollama](https://ollama.com/library)."
            ),
            pt=(
                "Nome exato do modelo, com sua tag, como mostrado por "
                "`ollama list` (por exemplo `llama3.2:3b` ou `qwen3:8b`). "
                "Deve ser baixado antes com `ollama pull <modelo>`. "
                "Veja a [biblioteca do Ollama](https://ollama.com/library)."
            ),
            de=(
                "Exakter Modellname mit Tag, wie in `ollama list` angezeigt "
                "(z. B. `llama3.2:3b` oder `qwen3:8b`). "
                "Muss zuvor mit `ollama pull <Modell>` geladen werden. "
                "Siehe die [Ollama-Bibliothek](https://ollama.com/library)."
            ),
            zh=(
                "模型的确切名称（含标签），与 `ollama list` 中显示的一致，"
                "例如 `llama3.2:3b` 或 `qwen3:8b`。"
                "必须先用 `ollama pull <模型>` 下载。"
                "浏览 [Ollama 模型库](https://ollama.com/library)。"
            ),
        ),
        alias=MultilingualString(
            en="Model",
            es="Modelo",
            pt="Modelo",
            de="Modell",
            zh="模型",
        ),
    )  # ty: ignore[invalid-type-form]

    max_tokens: schema_field(
        int_field(ge=1),
        placeholder=DEFAULT_MAX_TOKENS,
        description=MultilingualString(
            en=(
                "Maximum number of new tokens generated per response "
                "(`num_predict`). About 1 token is 0.75 English words.\n\n"
                "- **256-1024**: short answers\n"
                "- **2048-4096**: detailed answers or code\n"
                "- **4096 or more**: reasoning models, which also spend tokens "
                "on hidden thinking (the answer is cut off otherwise)\n\n"
                "Together with the prompt it must fit in the context window."
            ),
            es=(
                "Número máximo de tokens nuevos generados por respuesta "
                "(`num_predict`). Aproximadamente 1 token son 0.75 palabras.\n\n"
                "- **256-1024**: respuestas cortas\n"
                "- **2048-4096**: respuestas detalladas o código\n"
                "- **4096 o más**: modelos de razonamiento, que también gastan "
                "tokens en su pensamiento oculto (si no, la respuesta se "
                "corta)\n\n"
                "Junto con el prompt debe caber en la ventana de contexto."
            ),
            pt=(
                "Número máximo de tokens novos gerados por resposta "
                "(`num_predict`). Aproximadamente 1 token equivale a 0.75 "
                "palavras.\n\n"
                "- **256-1024**: respostas curtas\n"
                "- **2048-4096**: respostas detalhadas ou código\n"
                "- **4096 ou mais**: modelos de raciocínio, que também gastam "
                "tokens no pensamento oculto (senão a resposta é cortada)\n\n"
                "Junto com o prompt deve caber na janela de contexto."
            ),
            de=(
                "Maximale Anzahl neuer Token pro Antwort (`num_predict`). Etwa "
                "1 Token entspricht 0,75 Wörtern.\n\n"
                "- **256-1024**: kurze Antworten\n"
                "- **2048-4096**: ausführliche Antworten oder Code\n"
                "- **4096 oder mehr**: Reasoning-Modelle, die auch Token für "
                "verborgenes Denken verbrauchen (sonst wird die Antwort "
                "abgeschnitten)\n\n"
                "Zusammen mit dem Prompt muss es ins Kontextfenster passen."
            ),
            zh=(
                "每次响应生成的最大新 token 数量（`num_predict`）。"
                "约 1 个 token 相当于 0.75 个英文单词。\n\n"
                "- **256-1024**：简短回答\n"
                "- **2048-4096**：详细回答或代码\n"
                "- **4096 或更高**：推理模型，其隐藏思考也会消耗 token"
                "（否则回答会被截断）\n\n"
                "它与提示词一起必须不超过上下文窗口。"
            ),
        ),
        alias=MultilingualString(
            en="Max tokens",
            es="Tokens máximos",
            pt="Tokens máximos",
            de="Maximale neue Token",
            zh="最大 token 数",
        ),
    )  # ty: ignore[invalid-type-form]

    temperature: schema_field(
        float_field(ge=0.0, le=2.0),
        placeholder=DEFAULT_TEMPERATURE,
        description=MultilingualString(
            en=(
                "Controls how random the output is (range 0.0-2.0).\n\n"
                "- **0.0-0.3**: factual answers, extraction and RAG\n"
                "- **0.6-0.8**: general chat (recommended by most current "
                "models)\n"
                "- **1.0 or more**: creative writing\n\n"
                "At 0.0 the output is nearly deterministic."
            ),
            es=(
                "Controla qué tan aleatoria es la salida (rango 0.0-2.0).\n\n"
                "- **0.0-0.3**: respuestas factuales, extracción y RAG\n"
                "- **0.6-0.8**: chat general (lo recomendado por la mayoría de "
                "los modelos actuales)\n"
                "- **1.0 o más**: escritura creativa\n\n"
                "En 0.0 la salida es casi determinista."
            ),
            pt=(
                "Controla quão aleatória é a saída (intervalo 0.0-2.0).\n\n"
                "- **0.0-0.3**: respostas factuais, extração e RAG\n"
                "- **0.6-0.8**: chat geral (recomendado pela maioria dos "
                "modelos atuais)\n"
                "- **1.0 ou mais**: escrita criativa\n\n"
                "Em 0.0 a saída é quase determinística."
            ),
            de=(
                "Steuert, wie zufällig die Ausgabe ist (0.0-2.0).\n\n"
                "- **0.0-0.3**: Fakten, Extraktion und RAG\n"
                "- **0.6-0.8**: allgemeiner Chat (von den meisten aktuellen "
                "Modellen empfohlen)\n"
                "- **1.0 oder mehr**: kreatives Schreiben\n\n"
                "Bei 0.0 ist die Ausgabe nahezu deterministisch."
            ),
            zh=(
                "控制输出的随机程度（范围 0.0-2.0）。\n\n"
                "- **0.0-0.3**：事实性回答、信息抽取和 RAG\n"
                "- **0.6-0.8**：通用聊天（多数当前模型的推荐值）\n"
                "- **1.0 或更高**：创意写作\n\n"
                "0.0 时输出几乎是确定性的。"
            ),
        ),
        alias=MultilingualString(
            en="Temperature",
            es="Temperatura",
            pt="Temperatura",
            de="Temperatur",
            zh="温度",
        ),
    )  # ty: ignore[invalid-type-form]

    top_p: schema_field(
        float_field(ge=0.0, le=1.0),
        placeholder=DEFAULT_TOP_P,
        description=MultilingualString(
            en=(
                "Nucleus sampling (range 0.0-1.0): only the most likely tokens "
                "whose probabilities add up to this value are considered.\n\n"
                "- **0.8-0.95**: usual range (0.9 is a safe default)\n"
                "- **Lower**: more focused and predictable\n"
                "- **1.0**: no filtering\n\n"
                "Change either this or the temperature, not both at once."
            ),
            es=(
                "Muestreo por núcleo (rango 0.0-1.0): solo se consideran los "
                "tokens más probables cuyas probabilidades suman este valor.\n\n"
                "- **0.8-0.95**: rango habitual (0.9 es un valor seguro)\n"
                "- **Más bajo**: más enfocado y predecible\n"
                "- **1.0**: sin filtrado\n\n"
                "Cambie este valor o la temperatura, no ambos a la vez."
            ),
            pt=(
                "Amostragem por núcleo (intervalo 0.0-1.0): só os tokens mais "
                "prováveis cujas probabilidades somam este valor são "
                "considerados.\n\n"
                "- **0.8-0.95**: faixa habitual (0.9 é um valor seguro)\n"
                "- **Menor**: mais focado e previsível\n"
                "- **1.0**: sem filtragem\n\n"
                "Altere este valor ou a temperatura, não ambos ao mesmo tempo."
            ),
            de=(
                "Nucleus-Sampling (0.0-1.0): Es werden nur die wahrscheinlichsten "
                "Token berücksichtigt, deren Wahrscheinlichkeiten zusammen diesen "
                "Wert ergeben.\n\n"
                "- **0.8-0.95**: üblicher Bereich (0.9 ist ein sicherer Standard)\n"
                "- **Niedriger**: fokussierter und vorhersehbarer\n"
                "- **1.0**: keine Filterung\n\n"
                "Ändern Sie entweder diesen Wert oder die Temperatur, nicht "
                "beide gleichzeitig."
            ),
            zh=(
                "核采样（范围 0.0-1.0）：只考虑累计概率达到该值的最可能的 token。"
                "\n\n"
                "- **0.8-0.95**：常用范围（0.9 是稳妥的默认值）\n"
                "- **更低**：更集中、更可预测\n"
                "- **1.0**：不过滤\n\n"
                "请只调整该值或温度之一，不要同时调整。"
            ),
        ),
        alias=MultilingualString(
            en="Top P",
            es="Top P",
            pt="Top P",
            de="Top P",
            zh="Top P",
        ),
    )  # ty: ignore[invalid-type-form]

    top_k: schema_field(
        int_field(ge=0),
        placeholder=DEFAULT_TOP_K,
        description=MultilingualString(
            en=(
                "Only the K most likely tokens are considered at each step.\n\n"
                "- **20-50**: usual range (40 is the Ollama default)\n"
                "- **Lower**: more conservative output\n"
                "- **0**: disabled (no limit)"
            ),
            es=(
                "En cada paso solo se consideran los K tokens más probables."
                "\n\n"
                "- **20-50**: rango habitual (40 es el valor por defecto de "
                "Ollama)\n"
                "- **Más bajo**: salida más conservadora\n"
                "- **0**: desactivado (sin límite)"
            ),
            pt=(
                "Em cada passo só os K tokens mais prováveis são considerados."
                "\n\n"
                "- **20-50**: faixa habitual (40 é o padrão do Ollama)\n"
                "- **Menor**: saída mais conservadora\n"
                "- **0**: desativado (sem limite)"
            ),
            de=(
                "In jedem Schritt werden nur die K wahrscheinlichsten Token "
                "berücksichtigt.\n\n"
                "- **20-50**: üblicher Bereich (40 ist der Ollama-Standard)\n"
                "- **Niedriger**: konservativere Ausgabe\n"
                "- **0**: deaktiviert (keine Begrenzung)"
            ),
            zh=(
                "每一步只考虑概率最高的 K 个 token。\n\n"
                "- **20-50**：常用范围（40 是 Ollama 的默认值）\n"
                "- **更低**：输出更保守\n"
                "- **0**：禁用（不限制）"
            ),
        ),
        alias=MultilingualString(
            en="Top K",
            es="Top K",
            pt="Top K",
            de="Top K",
            zh="Top K",
        ),
    )  # ty: ignore[invalid-type-form]

    context_window: schema_field(
        int_field(ge=1),
        placeholder=DEFAULT_CONTEXT_WINDOW,
        description=MultilingualString(
            en=(
                "Total tokens the model can keep in memory (`num_ctx`): the "
                "conversation history, any retrieved documents and the "
                "response.\n\n"
                "- **8192**: short chats on low-memory computers\n"
                "- **16384**: chat and RAG on most computers (default)\n"
                "- **32768 or more**: long documents; needs a lot more RAM or "
                "VRAM\n\n"
                "Memory use grows with this value, and what exceeds it is "
                "silently dropped. Do not exceed what the model supports."
            ),
            es=(
                "Total de tokens que el modelo puede mantener en memoria "
                "(`num_ctx`): el historial de la conversación, los documentos "
                "recuperados y la respuesta.\n\n"
                "- **8192**: chats cortos en computadores con poca memoria\n"
                "- **16384**: chat y RAG en la mayoría de los computadores "
                "(por defecto)\n"
                "- **32768 o más**: documentos largos; requiere bastante más "
                "RAM o VRAM\n\n"
                "El uso de memoria crece con este valor, y lo que lo excede se "
                "descarta sin aviso. No supere lo que el modelo soporta."
            ),
            pt=(
                "Total de tokens que o modelo pode manter na memória "
                "(`num_ctx`): o histórico da conversa, os documentos "
                "recuperados e a resposta.\n\n"
                "- **8192**: chats curtos em computadores com pouca memória\n"
                "- **16384**: chat e RAG na maioria dos computadores (padrão)\n"
                "- **32768 ou mais**: documentos longos; exige bem mais RAM ou "
                "VRAM\n\n"
                "O uso de memória cresce com este valor, e o que o excede é "
                "descartado sem aviso. Não ultrapasse o que o modelo suporta."
            ),
            de=(
                "Gesamtzahl der Token, die das Modell im Speicher halten kann "
                "(`num_ctx`): Gesprächsverlauf, abgerufene Dokumente und "
                "Antwort.\n\n"
                "- **8192**: kurze Chats auf Rechnern mit wenig Speicher\n"
                "- **16384**: Chat und RAG auf den meisten Rechnern (Standard)\n"
                "- **32768 oder mehr**: lange Dokumente; benötigt deutlich mehr "
                "RAM oder VRAM\n\n"
                "Der Speicherbedarf wächst mit diesem Wert, und Überschüssiges "
                "wird ohne Hinweis verworfen. Nicht mehr als vom Modell "
                "unterstützt."
            ),
            zh=(
                "模型可在内存中保留的 token 总数（`num_ctx`），包括对话历史、"
                "检索到的文档和响应。\n\n"
                "- **8192**：内存较小的电脑上的短对话\n"
                "- **16384**：大多数电脑上的聊天和 RAG（默认）\n"
                "- **32768 或更高**：长文档；需要多得多的内存或显存\n\n"
                "内存占用随该值增长，超出部分会被静默丢弃。"
                "不要超过模型支持的上限。"
            ),
        ),
        alias=MultilingualString(
            en="Context window",
            es="Ventana de contexto",
            pt="Janela de contexto",
            de="Kontextfenster",
            zh="上下文窗口",
        ),
    )  # ty: ignore[invalid-type-form]

    think: schema_field(
        enum_field(
            enum=["auto", "on", "off"],
            labels={
                "auto": MultilingualString(
                    en="Auto",
                    es="Automático",
                    pt="Automático",
                    de="Automatisch",
                    zh="自动",
                ),
                "on": MultilingualString(
                    en="On",
                    es="Activado",
                    pt="Ativado",
                    de="An",
                    zh="开启",
                ),
                "off": MultilingualString(
                    en="Off",
                    es="Desactivado",
                    pt="Desativado",
                    de="Aus",
                    zh="关闭",
                ),
            },
        ),
        placeholder=DEFAULT_THINK,
        description=MultilingualString(
            en=(
                "Whether the model reasons step by step before answering. Only "
                "for models with the `thinking` capability (see "
                "`ollama show <model>`).\n\n"
                "- **Auto**: use the model's own default\n"
                "- **On**: better on hard problems, but slower and uses many "
                "tokens\n"
                "- **Off**: fast answers; avoids cut-off replies with a small "
                "Max tokens\n\n"
                "On or Off fail with models that cannot think; leave it on "
                "Auto for those."
            ),
            es=(
                "Si el modelo razona paso a paso antes de responder. Solo para "
                "modelos con la capacidad `thinking` (vea `ollama show "
                "<modelo>`).\n\n"
                "- **Automático**: usa el comportamiento propio del modelo\n"
                "- **Activado**: mejor en problemas difíciles, pero más lento "
                "y gasta muchos tokens\n"
                "- **Desactivado**: respuestas rápidas; evita respuestas "
                "cortadas con pocos Tokens máximos\n\n"
                "Activado o Desactivado fallan en modelos que no razonan; "
                "déjelo en Automático para esos."
            ),
            pt=(
                "Se o modelo raciocina passo a passo antes de responder. Só "
                "para modelos com a capacidade `thinking` (veja `ollama show "
                "<modelo>`).\n\n"
                "- **Automático**: usa o comportamento próprio do modelo\n"
                "- **Ativado**: melhor em problemas difíceis, mas mais lento e "
                "gasta muitos tokens\n"
                "- **Desativado**: respostas rápidas; evita respostas cortadas "
                "com poucos Tokens máximos\n\n"
                "Ativado ou Desativado falham em modelos que não raciocinam; "
                "deixe em Automático para esses."
            ),
            de=(
                "Ob das Modell vor der Antwort Schritt für Schritt überlegt. "
                "Nur für Modelle mit der Fähigkeit `thinking` (siehe `ollama "
                "show <Modell>`).\n\n"
                "- **Automatisch**: Standardverhalten des Modells\n"
                "- **An**: besser bei schwierigen Aufgaben, aber langsamer und "
                "viele Token\n"
                "- **Aus**: schnelle Antworten; vermeidet abgeschnittene "
                "Antworten bei kleinen Maximalen neuen Token\n\n"
                "An oder Aus schlagen bei Modellen ohne Denkfunktion fehl; "
                "dort Automatisch lassen."
            ),
            zh=(
                "模型在回答前是否逐步推理。仅适用于具有 `thinking` 能力的模型"
                "（见 `ollama show <模型>`）。\n\n"
                "- **自动**：使用模型自身的默认行为\n"
                "- **开启**：难题效果更好，但更慢且消耗大量 token\n"
                "- **关闭**：回答更快；在最大 token 数较小时避免回答被截断\n\n"
                "对不支持推理的模型，开启或关闭会报错；请保持“自动”。"
            ),
        ),
        alias=MultilingualString(
            en="Thinking",
            es="Razonamiento",
            pt="Raciocínio",
            de="Denkmodus",
            zh="推理",
        ),
    )  # ty: ignore[invalid-type-form]

    keep_alive: schema_field(
        string_field(),
        placeholder=DEFAULT_KEEP_ALIVE,
        description=MultilingualString(
            en=(
                "How long Ollama keeps the model loaded in memory after the "
                "last request. A loaded model answers faster; an unloaded one "
                "frees RAM or VRAM.\n\n"
                "- **`5m`**: Ollama default\n"
                "- **`30m`, `1h`**: keep it ready during a work session\n"
                "- **`-1`**: never unload\n"
                "- **`0`**: unload right after each answer"
            ),
            es=(
                "Cuánto tiempo Ollama mantiene el modelo cargado en memoria "
                "tras la última solicitud. Un modelo cargado responde más "
                "rápido; uno descargado libera RAM o VRAM.\n\n"
                "- **`5m`**: valor por defecto de Ollama\n"
                "- **`30m`, `1h`**: mantenerlo listo durante una sesión de "
                "trabajo\n"
                "- **`-1`**: nunca descargarlo\n"
                "- **`0`**: descargarlo justo después de cada respuesta"
            ),
            pt=(
                "Quanto tempo o Ollama mantém o modelo carregado na memória "
                "após a última solicitação. Um modelo carregado responde mais "
                "rápido; um descarregado libera RAM ou VRAM.\n\n"
                "- **`5m`**: padrão do Ollama\n"
                "- **`30m`, `1h`**: mantê-lo pronto durante uma sessão de "
                "trabalho\n"
                "- **`-1`**: nunca descarregar\n"
                "- **`0`**: descarregar logo após cada resposta"
            ),
            de=(
                "Wie lange Ollama das Modell nach der letzten Anfrage im "
                "Speicher hält. Ein geladenes Modell antwortet schneller; ein "
                "entladenes gibt RAM oder VRAM frei.\n\n"
                "- **`5m`**: Ollama-Standard\n"
                "- **`30m`, `1h`**: während einer Arbeitssitzung bereithalten\n"
                "- **`-1`**: nie entladen\n"
                "- **`0`**: direkt nach jeder Antwort entladen"
            ),
            zh=(
                "最后一次请求后，Ollama 将模型保留在内存中的时长。已加载的模型"
                "响应更快；卸载后可释放内存或显存。\n\n"
                "- **`5m`**：Ollama 默认值\n"
                "- **`30m`、`1h`**：在工作期间保持就绪\n"
                "- **`-1`**：永不卸载\n"
                "- **`0`**：每次回答后立即卸载"
            ),
        ),
        alias=MultilingualString(
            en="Keep alive",
            es="Mantener cargado",
            pt="Manter carregado",
            de="Geladen halten",
            zh="保持加载",
        ),
    )  # ty: ignore[invalid-type-form]

    timeout: schema_field(
        int_field(ge=1),
        placeholder=DEFAULT_TIMEOUT,
        description=MultilingualString(
            en=(
                "Seconds to wait for the server before giving up. The first "
                "request after a model is downloaded or unloaded is slower "
                "because Ollama must load it into memory. Replies are not "
                "streamed, so large or reasoning models on CPU may need 600 "
                "seconds or more."
            ),
            es=(
                "Segundos de espera al servidor antes de rendirse. La primera "
                "solicitud tras descargar o descargar de memoria un modelo es "
                "más lenta porque Ollama debe cargarlo. Las respuestas no se "
                "transmiten en streaming, así que los modelos grandes o de "
                "razonamiento en CPU pueden necesitar 600 segundos o más."
            ),
            pt=(
                "Segundos de espera pelo servidor antes de desistir. A "
                "primeira solicitação após baixar ou descarregar um modelo é "
                "mais lenta porque o Ollama precisa carregá-lo. As respostas "
                "não são transmitidas em streaming, então modelos grandes ou "
                "de raciocínio em CPU podem precisar de 600 segundos ou mais."
            ),
            de=(
                "Sekunden Wartezeit auf den Server, bevor abgebrochen wird. "
                "Die erste Anfrage nach dem Laden oder Entladen eines Modells "
                "ist langsamer, da Ollama es in den Speicher laden muss. "
                "Antworten werden nicht gestreamt; große oder Reasoning-Modelle "
                "auf der CPU brauchen ggf. 600 Sekunden oder mehr."
            ),
            zh=(
                "放弃前等待服务器的秒数。模型刚下载或被卸载后，第一次请求会更慢，"
                "因为 Ollama 需要把它加载到内存。回复不是流式输出，"
                "因此 CPU 上的大模型或推理模型可能需要 600 秒或更久。"
            ),
        ),
        alias=MultilingualString(
            en="Timeout (s)",
            es="Tiempo máximo (s)",
            pt="Tempo limite (s)",
            de="Zeitlimit (s)",
            zh="超时（秒）",
        ),
    )  # ty: ignore[invalid-type-form]


class OllamaModel(TextToTextGenerationTaskModel):
    """Chat model served by a local Ollama instance.

    The weights live in Ollama, not in DashAI, so this component has no
    download step: pull the model with ``ollama pull`` first. Requires the
    ``dashai-ollama`` plugin (``pip install dashai-ollama``).
    """

    SCHEMA = OllamaSchema
    COLOR: str = "#455a64"
    DISPLAY_NAME = MultilingualString(
        en="Ollama Model",
        es="Modelo Ollama",
        pt="Modelo Ollama",
        de="Ollama-Modell",
        zh="Ollama 模型",
    )
    DESCRIPTION = MultilingualString(
        en=(
            "Chat with any model served by a local Ollama server, or use it as "
            "the generation model of a RAG pipeline. Setup: 1) Install Ollama "
            "from [ollama.com/download](https://ollama.com/download). 2) Start "
            "it with `ollama serve` (it usually starts by itself after "
            "installing). 3) Download a model with `ollama pull <model>`, for "
            "example `ollama pull bonsai-2:9b`; browse the [Ollama "
            "library](https://ollama.com/library). 4) Type the exact name shown "
            "by `ollama list` in the Model field. Reasoning models need a "
            "higher Max tokens value. Source code and full documentation: "
            "[dashai-ollama on "
            "GitHub](https://github.com/idia-tt/dashai-ollama)."
        ),
        es=(
            "Chatee con cualquier modelo servido por un servidor Ollama local, "
            "o úselo como modelo de generación de un pipeline RAG. "
            "Configuración: 1) Instale Ollama desde "
            "[ollama.com/download](https://ollama.com/download). 2) Inícielo "
            "con `ollama serve` (normalmente se inicia solo tras la "
            "instalación). 3) Descargue un modelo con `ollama pull <modelo>`, "
            "por ejemplo `ollama pull bonsai-2:9b`; explore la [biblioteca de "
            "Ollama](https://ollama.com/library). 4) Escriba en el campo Modelo "
            "el nombre exacto que muestra `ollama list`. Los modelos de "
            "razonamiento necesitan un valor mayor de Tokens máximos. Código "
            "fuente y documentación completa: [dashai-ollama en "
            "GitHub](https://github.com/idia-tt/dashai-ollama)."
        ),
        pt=(
            "Converse com qualquer modelo servido por um servidor Ollama local, "
            "ou use-o como modelo de geração de um pipeline RAG. Configuração: "
            "1) Instale o Ollama em "
            "[ollama.com/download](https://ollama.com/download). 2) Inicie-o "
            "com `ollama serve` (normalmente ele inicia sozinho após a "
            "instalação). 3) Baixe um modelo com `ollama pull <modelo>`, por "
            "exemplo `ollama pull bonsai-2:9b`; veja a [biblioteca do "
            "Ollama](https://ollama.com/library). 4) Digite no campo Modelo o "
            "nome exato mostrado por `ollama list`. Modelos de raciocínio "
            "precisam de um valor maior de Tokens máximos. Código-fonte e "
            "documentação completa: [dashai-ollama no "
            "GitHub](https://github.com/idia-tt/dashai-ollama)."
        ),
        de=(
            "Chatten Sie mit jedem Modell eines lokalen Ollama-Servers oder "
            "nutzen Sie es als Generierungsmodell einer RAG-Pipeline. "
            "Einrichtung: 1) Ollama von "
            "[ollama.com/download](https://ollama.com/download) installieren. "
            "2) Mit `ollama serve` starten (startet nach der Installation meist "
            "automatisch). 3) Ein Modell mit `ollama pull <Modell>` laden, z. "
            "B. `ollama pull bonsai-2:9b`; siehe die "
            "[Ollama-Bibliothek](https://ollama.com/library). 4) Im Feld Modell "
            "den genauen Namen eintragen, den `ollama list` anzeigt. "
            "Reasoning-Modelle benötigen einen höheren Wert für Maximale neue "
            "Token. Quellcode und vollständige Dokumentation: [dashai-ollama "
            "auf GitHub](https://github.com/idia-tt/dashai-ollama)."
        ),
        zh=(
            "与本地 Ollama 服务器提供的任意模型聊天，或将其用作 RAG 流水线的生成模"
            "型。设置步骤：1）从 [ollama.com/download](https://ollama.com/download)"
            " 安装 Ollama。2）用 `ollama serve` 启动（安装后通常会自动启动）。3）用"
            " `ollama pull <模型>` 下载模型，例如 `ollama pull bonsai-2:9b`；可浏览"
            " [Ollama 模型库](https://ollama.com/library)。4）在“模型”字段中输入 `o"
            "llama list` 显示的确切名称。推理模型需要更大的“最大 token 数”。源代码"
            "与完整文档：[GitHub 上的 dashai-ollama](https://github.com/idia-tt/das"
            "hai-ollama)。"
        ),
    )

    def __init__(self, **kwargs):
        """Create the client for the configured Ollama server."""
        kwargs = self.validate_and_transform(kwargs)
        self.host = kwargs.get("host", DEFAULT_HOST)
        self.model_name = kwargs["model"]
        self.options = {
            "num_predict": kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", DEFAULT_TOP_P),
            "top_k": kwargs.get("top_k", DEFAULT_TOP_K),
            "num_ctx": kwargs.get("context_window", DEFAULT_CONTEXT_WINDOW),
        }
        # "auto" sends nothing so each model keeps its own default.
        self.think = {"on": True, "off": False}.get(kwargs.get("think", DEFAULT_THINK))
        self.keep_alive = _parse_keep_alive(
            kwargs.get("keep_alive", DEFAULT_KEEP_ALIVE)
        )
        self._error_types = (ollama.ResponseError, ConnectionError, httpx.HTTPError)
        self.client = ollama.Client(
            host=self.host, timeout=kwargs.get("timeout", DEFAULT_TIMEOUT)
        )

    def generate(self, prompt: list) -> list[str]:
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
                think=self.think,
                keep_alive=self.keep_alive,
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
