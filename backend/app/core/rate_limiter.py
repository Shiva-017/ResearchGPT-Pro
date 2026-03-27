# backend/app/core/rate_limiter.py

"""
Multi-layer API protection:
  1. IP-based rate limiting (per hour + per day)
  2. Global daily budget cap (total queries across all users)
  3. Optional API key gating

All in-memory — resets on server restart. Good enough for a personal project.
"""

import time
from collections import defaultdict
from fastapi import Request, HTTPException
from backend.app.config import settings


# --- Layer 1: IP Rate Limiting ---

# {ip: [timestamp, timestamp, ...]}
_ip_requests: dict[str, list[float]] = defaultdict(list)  # keyed by client IP

# Limits (configurable via env)
IP_HOURLY_LIMIT = settings.rate_limit_per_ip_hourly    # e.g. 20
IP_DAILY_LIMIT = settings.rate_limit_per_ip_daily      # e.g. 50


def _cleanup(timestamps: list[float], window: float) -> list[float]:
    """Remove timestamps outside the window."""
    now = time.time()
    return [t for t in timestamps if now - t < window]


def check_ip_rate_limit(ip: str):
    """Raise 429 if IP exceeds hourly or daily limits."""
    now = time.time()

    # Clean old entries
    _ip_requests[ip] = _cleanup(_ip_requests[ip], 86400)  # keep 24h

    # Check hourly
    hour_reqs = [t for t in _ip_requests[ip] if now - t < 3600]
    if len(hour_reqs) >= IP_HOURLY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {IP_HOURLY_LIMIT} requests/hour maximum. Please try again later."
        )

    # Check daily
    if len(_ip_requests[ip]) >= IP_DAILY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit exceeded: {IP_DAILY_LIMIT} requests/day allowed. Quota resets at midnight UTC."
        )

    # Record this request
    _ip_requests[ip].append(now)


# --- Layer 2: Global Daily Budget Cap ---

_global_count = 0
_global_day = 0  # day number to track resets

GLOBAL_DAILY_LIMIT = settings.global_daily_query_limit  # e.g. 200


def check_global_limit():
    """Raise 429 if total queries today exceed budget cap."""
    global _global_count, _global_day

    today = int(time.time() // 86400)
    if today != _global_day:
        _global_day = today
        _global_count = 0

    if _global_count >= GLOBAL_DAILY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Service temporarily unavailable: daily query budget exhausted. Resets tomorrow."
        )

    _global_count += 1


# --- Layer 3: Optional API Key ---

def check_api_key(request: Request):
    """If API key protection is enabled, require valid key in header."""
    if not settings.require_api_key:
        return  # open access

    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if key not in settings.api_keys_set:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: invalid or missing API key. Pass it via X-API-Key header."
        )


# --- Combined middleware ---

def enforce_limits(request: Request):
    """Run all protection layers. Call this at the top of each endpoint."""
    # Layer 3: API key check (if enabled)
    check_api_key(request)

    # Layer 2: Global budget
    check_global_limit()

    # Layer 1: Per-IP rate limit
    ip = request.headers.get("X-Forwarded-For", request.client.host)
    if ip and "," in ip:
        ip = ip.split(",")[0].strip()  # first IP in chain
    check_ip_rate_limit(ip)
