"""
Startup check of the Langfuse credentials.

A wrong key/region makes the OpenTelemetry exporter log
"Failed to export spans batch code: 401" on every batch. Checking once at
startup turns that into one clear log line, and lets us switch tracing off
instead of failing on every request.
"""
import base64
import logging
from typing import Optional

import httpx

logger = logging.getLogger("trade_agent")

REGION_HOSTS = ["https://cloud.langfuse.com", "https://us.cloud.langfuse.com"]


def _status(host: str, public_key: str, secret_key: str) -> Optional[int]:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    try:
        r = httpx.get(
            f"{host.rstrip('/')}/api/public/projects",
            headers={"Authorization": f"Basic {token}"},
            timeout=10,
        )
        return r.status_code
    except httpx.HTTPError:
        return None  # network trouble: unknown, not a credentials verdict


def langfuse_credentials_ok(host: str, public_key: str, secret_key: str) -> bool:
    """False only when Langfuse positively rejects the keys (401/403)."""
    if not (public_key and secret_key):
        logger.warning("Langfuse keys not set: tracing is off.")
        return False
    status = _status(host, public_key, secret_key)
    if status not in (401, 403):
        return True
    others = [h for h in REGION_HOSTS if h.rstrip("/") != host.rstrip("/")]
    hint = ""
    for other in others:
        if _status(other, public_key, secret_key) == 200:
            hint = f" These keys DO work on {other}: set LANGFUSE_HOST to that."
            break
    logger.error(
        "Langfuse rejected the keys (HTTP %s) at %s: tracing disabled.%s "
        "Check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY belong to one project.",
        status, host, hint,
    )
    return False
