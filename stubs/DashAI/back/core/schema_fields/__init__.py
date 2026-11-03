"""Stand-in for ``DashAI.back.core.schema_fields``."""

from typing import Annotated, Any

from pydantic import BaseModel, Field


class BaseSchema(BaseModel):
    """Base class of every component schema."""


def schema_field(t: Any, placeholder: Any, description: Any, alias: Any = None) -> Any:
    return Annotated[
        t,
        Field(
            description=str(description),
            json_schema_extra={"placeholder": placeholder, "display_name": alias},
        ),
    ]


def int_field(ge: int | None = None, le: int | None = None) -> Any:
    return Annotated[int, Field(ge=ge, le=le)]


def float_field(ge: float | None = None, le: float | None = None) -> Any:
    return Annotated[float, Field(ge=ge, le=le)]


def string_field() -> Any:
    return str


def enum_field(enum: list[str], labels: dict[str, Any] | None = None) -> Any:
    return str
