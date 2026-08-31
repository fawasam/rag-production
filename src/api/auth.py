"""API-key authentication (SRS.md NFR: Security — API authentication).

Simple shared/per-client API key via the `X-API-Key` header, checked against
a comma-separated list in the API_KEYS env var. This is the minimum viable
auth for a production deployment — swap for OAuth/JWT if you need per-user
identity rather than per-client keys.

Uses secrets.compare_digest for the comparison so a mistyped/guessed key
can't be distinguished by response-time timing.
"""
import os
import secrets

from fastapi import Header, HTTPException, status


def _valid_keys() -> set[str]:
    raw = os.environ.get("API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def verify_api_key(x_api_key: str = Header(default="")) -> str:
    """FastAPI dependency — raises 401 if the key is missing or invalid,
    otherwise returns the key (useful for per-client logging/rate-limiting
    later)."""
    valid_keys = _valid_keys()

    if not valid_keys:
        # Fail closed: an unconfigured API_KEYS env var should block access,
        # not silently allow everyone through.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: API_KEYS is not set.",
        )

    if not x_api_key or not any(
        secrets.compare_digest(x_api_key, valid_key) for valid_key in valid_keys
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Pass it in the X-API-Key header.",
        )

    return x_api_key
