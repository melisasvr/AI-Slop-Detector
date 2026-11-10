"""Groq client factory + retry wrapper that turns SDK errors into friendly SlopErrors."""
from __future__ import annotations

import time
from typing import Callable, TypeVar

import groq
from groq import Groq

from .errors import ApiError, RateLimited

T = TypeVar("T")


def get_client(api_key: str) -> Groq:
    if not api_key:
        raise ApiError("Missing GROQ_API_KEY. Add it to .env or the sidebar.")
    return Groq(api_key=api_key, max_retries=0, timeout=90)  # we do our own retries


def _retry_after(e: groq.APIStatusError, default: float) -> float:
    try:
        return min(float(e.response.headers.get("retry-after", default)), 60.0)
    except (TypeError, ValueError):
        return default


def call_with_retry(fn: Callable[[], T], *, attempts: int = 4, on_wait: Callable[[str], None] | None = None) -> T:
    for i in range(attempts):
        last = i == attempts - 1
        backoff = 2.0 * (2 ** i)
        try:
            return fn()
        except groq.AuthenticationError as e:
            raise ApiError("Groq rejected the API key (401). Check GROQ_API_KEY.") from e
        except groq.RateLimitError as e:
            if last:
                raise RateLimited("Groq rate limit hit repeatedly. Wait a minute and retry.") from e
            wait = _retry_after(e, backoff)
            if on_wait:
                on_wait(f"Rate limited, retrying in {wait:.0f}s…")
            time.sleep(wait)
        except (groq.APIConnectionError, groq.InternalServerError) as e:  # timeouts subclass connection error
            if last:
                raise ApiError(f"Could not reach Groq: {e}") from e
            time.sleep(backoff)
        except groq.APIStatusError as e:
            raise ApiError(f"Groq API error {e.status_code}: {getattr(e, 'message', e)}") from e
    raise ApiError("unreachable")  # pragma: no cover
