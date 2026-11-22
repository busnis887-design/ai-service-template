"""FastAPI application factory with health, embeddings and grounded Q&A."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__
from .config import Settings
from .metrics import Metrics
from .schemas import (
    AskRequest,
    AskResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    HealthResponse,
    Source,
)
from .security import APIKeyAuth

logger = logging.getLogger("ai_service")

_TOKEN = re.compile(r"[a-z0-9]+")
_CORPUS = {
    "api-guide.txt": (
        "All endpoints require an API key in the X-API-Key header. "
        "Rate limiting is applied per key at 60 requests per minute. "
        "Failed inference calls use exponential backoff with three attempts."
    ),
    "architecture.txt": (
        "The service runs in a container behind a reverse proxy. "
        "Embeddings are indexed for cosine similarity search. "
        "Answers are grounded in retrieved context to reduce hallucination."
    ),
}


def embed(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    for token in _TOKEN.findall(text.lower()):
        index = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "big") % dim
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def create_app(
    settings: Settings | None = None, auth: APIKeyAuth | None = None
) -> FastAPI:
    settings = settings or Settings()
    auth = auth or APIKeyAuth(settings)
    app = FastAPI(title=settings.app_name, version=__version__)
    app.state.settings = settings
    app.state.auth = auth
    app.state.metrics = Metrics()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "X-Request-ID", "Content-Type"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        duration_ms = (time.perf_counter() - started) * 1000
        app.state.metrics.observe(request.url.path, response.status_code, duration_ms)
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                }
            )
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Invalid request payload"},
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/metrics", tags=["ops"])
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(app.state.metrics.render())

    @app.get("/v1/models", tags=["inference"])
    async def models(api_key: str = Depends(auth)) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": "hashing-v1", "owned_by": "local", "dim": settings.embedding_dim}
            ],
        }

    @app.post("/v1/embed", response_model=EmbedResponse, tags=["inference"])
    async def embed_text(
        payload: EmbedRequest,
        api_key: str = Depends(auth),
    ) -> EmbedResponse:
        vector = embed(payload.text, settings.embedding_dim)
        return EmbedResponse(
            model="hashing-v1", dim=settings.embedding_dim, vector=vector
        )

    @app.post("/v1/ask", response_model=AskResponse, tags=["inference"])
    async def ask(
        payload: AskRequest,
        api_key: str = Depends(auth),
    ) -> AskResponse:
        query_vector = embed(payload.question, settings.embedding_dim)
        ranked = sorted(
            (
                (name, _cosine(query_vector, embed(text, settings.embedding_dim)), text)
                for name, text in _CORPUS.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )[: payload.top_k]
        if not ranked or ranked[0][1] <= 0:
            return AskResponse(
                answer="I don't have enough information to answer that.", sources=[]
            )
        return AskResponse(
            answer=ranked[0][2],
            sources=[Source(document=name, score=round(score, 4)) for name, score, _ in ranked],
        )

    return app


app = create_app()
