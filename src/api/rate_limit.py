"""Per-API-key rate limiting (SRS.md NFR: Cost Control / Reliability).

A simple in-memory sliding-window limiter, chained after verify_api_key so
only authenticated requests get counted (and counted per-key, not globally).

Limitation, stated plainly: this is per-process, in-memory state. It works
correctly for a single uvicorn worker/single replica, which is what
docker-compose.yml runs. It does NOT share state across multiple workers or
horizontally-scaled replicas — that needs a shared store (Redis, most
commonly) keyed the same way. Swapping the storage backend is the only
change needed; the dependency's interface (raise 429 past the limit) stays
the same.
"""
import os
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, status

from src.api.auth import verify_api_key

RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", 30))
WINDOW_SECONDS = 60

_request_log: dict[str, deque] = defaultdict(deque)


def enforce_rate_limit(api_key: str = Depends(verify_api_key)) -> str:
    now = time.time()
    log = _request_log[api_key]

    while log and now - log[0] > WINDOW_SECONDS:
        log.popleft()

    if len(log) >= RATE_LIMIT_PER_MINUTE:
        retry_after = max(1, int(WINDOW_SECONDS - (now - log[0])) + 1)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {RATE_LIMIT_PER_MINUTE} requests/minute per API key.",
            headers={"Retry-After": str(retry_after)},
        )

    log.append(now)
    return api_key
