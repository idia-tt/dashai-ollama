"""Stand-in for ``DashAI.back.models.text_to_text_generation_model``."""

from typing import Any, ClassVar

from DashAI.back.core.schema_fields import BaseSchema


class TextToTextGenerationTaskModel:
    SCHEMA: ClassVar[type[BaseSchema]]
    COMPATIBLE_COMPONENTS: ClassVar[list[str]] = ["TextToTextGenerationTask", "RAGTask"]

    @classmethod
    def get_metadata(cls) -> dict[str, Any]:
        return {"requires_download": False, "download_size_bytes": None}

    def validate_and_transform(self, raw_data: dict) -> dict:
        return self.SCHEMA.model_validate(raw_data).model_dump()
