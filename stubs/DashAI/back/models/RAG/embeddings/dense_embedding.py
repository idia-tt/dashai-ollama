"""Stand-in for ``DashAI.back.models.RAG.embeddings.dense_embedding``."""

from typing import ClassVar

from DashAI.back.core.schema_fields import BaseSchema


class DenseEmbedding:
    SCHEMA: ClassVar[type[BaseSchema]]
    embedding_dim: int

    def validate_and_transform(self, raw_data: dict) -> dict:
        return self.SCHEMA.model_validate(raw_data).model_dump()
