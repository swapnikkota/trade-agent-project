"""
ChatGroq that retries Groq's intermittent "Parsing failed" errors.

Sometimes the model writes a malformed tool call and Groq answers with
`APIError: Parsing failed. The model generated output that could not be
parsed ... see 'failed_generation'`. It is non-deterministic: the same
request normally succeeds on a second try. Left alone, one such failure
(raised inside a sub-agent, as in production) aborts the whole chat turn.

Only retried when nothing has been streamed yet, so a retry can never
duplicate text the user already saw.
"""
import logging
from typing import Any, AsyncIterator, Iterator

import groq
from langchain_core.outputs import ChatGenerationChunk
from langchain_groq import ChatGroq

logger = logging.getLogger("trade_agent")

MAX_PARSE_RETRIES = 2
_MARKERS = (
    "parsing failed",
    "failed_generation",
    "tool_use_failed",
    "tool call validation failed",
    "failed to parse tool call",
)


def is_parse_failure(exc: BaseException) -> bool:
    if not isinstance(exc, groq.APIError):
        return False
    text = f"{getattr(exc, 'message', '')} {getattr(exc, 'body', '')}".lower()
    return any(m in text for m in _MARKERS)


def is_retryable(exc: BaseException) -> bool:
    """Parse failures, plus Groq's mid-stream `error` events.

    The SDK raises a bare `groq.APIError` (not an `APIStatusError`, so no HTTP
    status) when the stream itself carries an error event, e.g. a transient
    server-side failure. Those are not tied to our request being wrong, so
    they get the same pre-first-chunk retry.
    """
    return is_parse_failure(exc) or type(exc) is groq.APIError


class RetryingChatGroq(ChatGroq):
    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        attempt = 0
        while True:
            yielded = False
            try:
                async for chunk in super()._astream(*args, **kwargs):
                    yielded = True
                    yield chunk
                return
            except groq.APIError as exc:
                if yielded or attempt >= MAX_PARSE_RETRIES or not is_retryable(exc):
                    raise
                attempt += 1
                logger.warning(
                    "Groq stream error, retrying (%d/%d): %.300s", attempt, MAX_PARSE_RETRIES, exc
                )

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        attempt = 0
        while True:
            yielded = False
            try:
                for chunk in super()._stream(*args, **kwargs):
                    yielded = True
                    yield chunk
                return
            except groq.APIError as exc:
                if yielded or attempt >= MAX_PARSE_RETRIES or not is_retryable(exc):
                    raise
                attempt += 1
                logger.warning(
                    "Groq stream error, retrying (%d/%d): %.300s", attempt, MAX_PARSE_RETRIES, exc
                )
