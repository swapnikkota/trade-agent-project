"""User-facing error text for failed chat turns."""
import re

_HTML = re.compile(r"<\s*(!doctype|html)\b", re.I)
MAX_LEN = 300


def friendly_error(exc: BaseException) -> str:
    """
    Short, readable message for the chat UI. Full detail (including any raw
    HTML error page from a proxy/gateway) stays in the server log only.
    """
    text = str(exc)
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__.lower()

    if status in (502, 503, 504) or _HTML.search(text):
        return (
            "The AI service is temporarily unavailable (it may be waking up). "
            "Please try again in a few seconds."
        )
    if status == 429 or "rate limit" in text.lower():
        return "The AI service is rate-limited right now. Please wait a moment and try again."
    if "connection" in name or "timeout" in name:
        return "Couldn't reach the AI service. Please try again in a few seconds."
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN].rstrip() + "…"
    return text
