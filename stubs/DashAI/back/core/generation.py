"""Stand-in for ``DashAI.back.core.generation`` (typed text generation).

Only the names the plugin uses, with plain dataclasses instead of the host's
Pydantic models. Hosts that predate this module do not have it at all, which is
why the plugin imports it inside ``generate_result``.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextPart:
    text: str
    type: str = "text"


@dataclass(frozen=True)
class ReasoningPart:
    text: str
    type: str = "reasoning"


@dataclass(frozen=True)
class GenerationUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    generation_seconds: float | None = None
    load_seconds: float | None = None


@dataclass(frozen=True)
class GenerationResult:
    parts: list[TextPart | ReasoningPart]
    usage: GenerationUsage | None = None
    finish_reason: str | None = None

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[dict[str, str]] = field(default_factory=list)

    def to_messages(self) -> list[dict[str, str]]:
        return list(self.messages)
