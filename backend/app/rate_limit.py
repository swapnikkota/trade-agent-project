"""
Shared slowapi Limiter instance (Phase 6).

Lives in its own module, separate from main.py, so that app/api/threads.py
can import it too without creating a circular import (main.py imports
threads_router from app.api.threads; if threads.py imported the limiter
back from app.main, that import would cycle).
"""
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

API_KEY_HEADER = "X-API-Key"


def rate_limit_key(request: Request) -> str:
    """
    Rate-limit per API key when auth is configured, so the limit tracks
    identity rather than network address (holds steady across NATs/shared
    IPs, and gives each key its own budget). Falls back to remote address
    when no key is configured.
    """
    api_key = request.headers.get(API_KEY_HEADER)
    return api_key or get_remote_address(request)


# Phase 6: rate limiting. Caps requests per key/IP so a runaway loop (bug
# or otherwise) can't hammer Ollama/Anthropic or flood Langfuse with
# traces. /chat is capped tighter since it's by far the most expensive
# call; /threads endpoints get a looser cap since they're cheap DB reads.
limiter = Limiter(key_func=rate_limit_key)
