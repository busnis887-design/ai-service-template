# AI Service Template

[![tests](https://github.com/busnis887-design/ai-service-template/actions/workflows/tests.yml/badge.svg)](https://github.com/busnis887-design/ai-service-template/actions/workflows/tests.yml) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-minded **FastAPI** template for AI inference endpoints: API-key auth, per-key rate limiting, request IDs, structured logs, validation errors, health checks, and a grounded `/v1/ask` endpoint. Runs locally with no external services or API keys.

## Endpoints

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/health` | no | Liveness + version |
| POST | `/v1/embed` | `X-API-Key` | Text → normalized vector |
| POST | `/v1/ask` | `X-API-Key` | Grounded answer + ranked sources |

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
curl -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"question":"How does rate limiting work?"}' http://localhost:8000/v1/ask
```

Docker:

```bash
docker build -t ai-service-template .
docker run -p 8000:8000 -e AI_SERVICE_API_KEYS="prod-key" ai-service-template
```

## Configuration

All settings use the `AI_SERVICE_` prefix and come from the environment:

| Variable | Default | Meaning |
| --- | --- | --- |
| `AI_SERVICE_API_KEYS` | `dev-key,local-test-key` | Comma-separated valid keys |
| `AI_SERVICE_RATE_LIMIT_PER_MINUTE` | `60` | Per-key sliding window |
| `AI_SERVICE_EMBEDDING_DIM` | `256` | Hashing embedder dimension |
| `AI_SERVICE_APP_NAME` | `AI Service Template` | OpenAPI title |

## Design decisions

- **App factory (`create_app`)** makes tests independent: inject settings/auth instead of mutating globals.
- **Sliding-window rate limiter** keyed by API key; 429 with a clear detail message.
- **Request ID middleware** echoes `X-Request-ID` and emits one JSON log line per request with status and duration.
- **Grounded answers.** `/v1/ask` only returns text from the local corpus and always lists sources with cosine scores.
- **Swappable inference.** Replace the hashing embedder and local corpus with your model/vector DB; the API contract stays the same.

## Tests

```bash
python -m unittest discover -s tests -v
```

Covers public health, missing/invalid keys, embedding shape, grounded answers with citations, 422 validation, 429 rate limiting, and vector normalization.

## License

MIT
