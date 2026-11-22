"""API key authentication and per-key sliding-window rate limiting."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from .config import Settings


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        hits.append(now)


class APIKeyAuth:
    def __init__(self, settings: Settings) -> None:
        self.keys = settings.key_set
        self.limiter = RateLimiter(settings.rate_limit_per_minute)

    def __call__(self, request: Request) -> str:
        key = request.headers.get("X-API-Key", "")
        if key not in self.keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key",
            )
        self.limiter.check(key)
        return key
