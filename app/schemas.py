"""Request and response models."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str


class EmbedRequest(BaseModel):
    text: str = Field(min_length=1, description="Text to embed")


class EmbedResponse(BaseModel):
    model: str
    dim: int
    vector: list[float]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    document: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class ErrorResponse(BaseModel):
    detail: str
