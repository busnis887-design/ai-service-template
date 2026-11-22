"""Minimal in-process Prometheus-style metrics (no external dependencies)."""

from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[str, int] = defaultdict(int)
        self._status: dict[str, int] = defaultdict(int)
        self._duration: dict[str, float] = defaultdict(float)

    def observe(self, path: str, status: int, duration_ms: float) -> None:
        with self._lock:
            self._requests[path] += 1
            self._status[f"{status}"] += 1
            self._duration[path] += duration_ms

    def render(self) -> str:
        with self._lock:
            lines = ["# HELP http_requests_total Total HTTP requests by path", "# TYPE http_requests_total counter"]
            for path, count in sorted(self._requests.items()):
                lines.append(f'http_requests_total{{path="{path}"}} {count}')
            lines += ["# HELP http_request_duration_ms_sum Total request duration", "# TYPE http_request_duration_ms_sum counter"]
            for path, total in sorted(self._duration.items()):
                lines.append(f'http_request_duration_ms_sum{{path="{path}"}} {total:.2f}')
            lines += ["# HELP http_responses_total Responses by status code", "# TYPE http_responses_total counter"]
            for status, count in sorted(self._status.items()):
                lines.append(f'http_responses_total{{status="{status}"}} {count}')
        return "\n".join(lines) + "\n"
