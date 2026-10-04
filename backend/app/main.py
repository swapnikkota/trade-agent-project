"""
FastAPI entrypoint.

Phase 1: /chat endpoint (streaming), agent wired to the trade MCP server,
Langfuse tracing per turn.

Phase 2 (this revision): Postgres-backed threads. /chat now requires a
thread_id, loads a sliding window of prior messages as context, and
persists both the user's message and the agent's final reply after each
turn. /threads/* (app/api/threads.py) handles thread CRUD for the
frontend's conversation sidebar.
"""
import json
import logging
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Must run before app.config / langfuse import anything that reads env vars,
# since langfuse v3's client reads LANGFUSE_* directly from os.environ.
load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.agent.factory import build_agent, build_langfuse_handler
from app.api.threads import router as threads_router
from app.config import get_settings
from app.db import repository
from app.db.session import create_all_tables, dispose_db, get_db, init_db
from app.mcp.registry import get_mcp_tools
from app.rate_limit import API_KEY_HEADER, limiter

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger("trade_agent")

_state = {"agent": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    active_model = {
        "ollama": settings.ollama_model,
        "anthropic": settings.anthropic_model,
        "groq": settings.groq_model,
    }[settings.model_provider]
    logger.info("Model provider: %s | Model: %s", settings.model_provider, active_model)

    logger.info("Connecting to Postgres and ensuring tables exist...")
    init_db(settings)
    await create_all_tables()

    logger.info("Connecting to MCP servers...")
    tools = await get_mcp_tools(settings)
    logger.info("Loaded %d MCP tool(s): %s", len(tools), [t.name for t in tools])
    _state["agent"] = build_agent(tools, settings)

    yield

    await dispose_db()
    _state.clear()


_PUBLIC_PATHS = {"/health"}

app = FastAPI(title="Trade Agent Backend", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """
    Phase 6: shared-secret auth. If settings.api_key is unset, this is a
    no-op (local dev default). When set, every request except /health and
    CORS preflight OPTIONS requests must send a matching X-API-Key header.

    OPTIONS is always let through regardless of the key, since browsers
    send CORS preflight requests without custom headers by design — CORS
    middleware (registered separately) is what actually answers those.
    """
    settings = get_settings()
    if not settings.api_key or request.method == "OPTIONS" or request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    provided = request.headers.get(API_KEY_HEADER)
    if provided != settings.api_key:
        return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key"})

    return await call_next(request)


app.include_router(threads_router)


class ChatRequest(BaseModel):
    message: str
    user_id: str
    thread_id: uuid.UUID


@app.get("/health")
async def health():
    return {"status": "ok"}


def _is_empty_tool_result(content: str) -> bool:
    """
    Best-effort detection of an empty / no-match tool result.

    This exists because prompt instructions alone did not stop the LLM
    (llama3.1:8b) from fabricating a plausible-looking trade after seeing
    an empty result — confirmed via a Langfuse trace where get_trades
    correctly returned [] and the model invented one anyway. An empty
    result is unambiguous, so we detect it here in code and short-circuit
    before the LLM gets another turn to narrate anything on top of it.

    Handles the realistic shapes get_trades / get_trade_by_id return: a
    bare empty list, an empty dict, JSON null, or a dict wrapping an
    empty list under a common key.
    """
    stripped = content.strip()
    if stripped in ("[]", "{}", "null", "None", ""):
        return True
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return False
    if parsed is None:
        return True
    if isinstance(parsed, list):
        return len(parsed) == 0
    if isinstance(parsed, dict):
        for key in ("trades", "results", "data", "items"):
            if key in parsed and isinstance(parsed[key], list):
                return len(parsed[key]) == 0
        return len(parsed) == 0
    return False


def _stringify_content(content) -> str:
    """
    LangGraph message content is sometimes a plain string, sometimes a list
    of content blocks (e.g. tool output: [{"type": "text", "text": "..."}]).
    Normalize to a plain string for storage.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


@app.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Streams the agent's response as Server-Sent Events.
    Each event is a JSON-encoded chunk: {"type": "token"|"done"|"error", "content": ...}

    On completion, persists the user's message and the agent's final reply
    to the thread so subsequent turns (and other devices/sessions) see the
    same history.
    """
    thread = await repository.get_thread(db, req.thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread.user_id != req.user_id:
        raise HTTPException(status_code=403, detail="Thread does not belong to this user")

    if not thread.title:
        # Auto-title the thread from its first message, so new threads
        # don't sit as "Untitled conversation" forever — the user can
        # still rename it manually afterward (double-click in the sidebar).
        auto_title = req.message.strip()
        if len(auto_title) > 60:
            auto_title = auto_title[:57].rstrip() + "…"
        if auto_title:
            await repository.rename_thread(db, req.thread_id, auto_title)

    settings = get_settings()
    agent = _state["agent"]
    langfuse_handler = build_langfuse_handler()

    history = await repository.get_recent_messages(db, req.thread_id, settings.history_window_size)
    agent_messages = [{"role": m.role, "content": m.content} for m in history]
    agent_messages.append({"role": "user", "content": req.message})

    # Persist the user's turn immediately, so it's saved even if the agent
    # run fails partway through.
    await repository.append_message(db, req.thread_id, role="user", content=req.message)

    async def event_generator():
        final_assistant_content = ""
        try:
            # stream_mode="messages" yields (message_chunk, metadata) pairs,
            # with the AI message arriving as a sequence of small token
            # chunks rather than one complete message at the end (that was
            # stream_mode="values", which only ever gave us the full final
            # text in one jump — fine for correctness, but meant the UI
            # couldn't show real token-by-token streaming). Tool messages
            # still arrive as one complete ToolMessage chunk either way, so
            # the empty-result short-circuit logic below is unchanged.
            async for message_chunk, metadata in agent.astream(
                {"messages": agent_messages},
                config={
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_user_id": req.user_id,
                        "langfuse_session_id": str(req.thread_id),
                    },
                },
                stream_mode="messages",
            ):
                msg_type = getattr(message_chunk, "type", None)
                logger.info(
                    "Stream chunk: msg_type=%s name=%s node=%s",
                    msg_type,
                    getattr(message_chunk, "name", None),
                    metadata.get("langgraph_node") if isinstance(metadata, dict) else None,
                )

                if msg_type == "human":
                    continue

                content = getattr(message_chunk, "content", None)

                if msg_type == "tool":
                    # A tool result (e.g. raw MCP response). Handled BEFORE
                    # the generic "no content" skip below, because an empty
                    # result (get_trades returning no rows) very often
                    # serializes to a falsy content value (e.g. ""), which
                    # would otherwise cause this chunk to be silently
                    # skipped entirely — leaving nothing to short-circuit
                    # on, and falling through to whatever the LLM's next
                    # turn happens to generate. That gap is exactly what
                    # let a fabricated trade slip through earlier even
                    # with this safeguard in place.
                    stringified = _stringify_content(content) if content is not None else ""
                    tool_name = getattr(message_chunk, "name", None) or "tool"
                    is_empty = _is_empty_tool_result(stringified)
                    logger.info(
                        "Tool result from %s (empty=%s): %.500s",
                        tool_name,
                        is_empty,
                        stringified,
                    )
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "tool_result", "tool": tool_name, "content": stringified}
                        ),
                    }

                    if is_empty:
                        logger.info("Short-circuiting: empty tool result from %s", tool_name)
                        # Structural safeguard (see _is_empty_tool_result):
                        # stop here rather than letting the graph continue
                        # to an LLM turn that summarizes "no data" — that's
                        # exactly the turn that has fabricated trades.
                        # Trade-off: if the agent legitimately intended a
                        # follow-up tool call after this one (e.g. trying a
                        # different symbol), that follow-up won't happen.
                        no_match_text = "No matching trades found."
                        final_assistant_content = no_match_text
                        yield {
                            "event": "message",
                            "data": json.dumps(
                                {"type": "assistant_delta", "content": no_match_text}
                            ),
                        }
                        break
                    continue

                if msg_type not in ("ai", "AIMessageChunk"):
                    # Anything else (e.g. internal middleware-injected
                    # messages) is not part of the visible reply.
                    #
                    # NOTE: stream_mode="messages" delivers each streamed
                    # token as an AIMessageChunk, whose `.type` attribute is
                    # literally the string "AIMessageChunk" — NOT "ai" the
                    # way a complete AIMessage is typed under
                    # stream_mode="values". An earlier version of this
                    # check only allowed "ai" and silently discarded every
                    # single streamed chunk as a result (confirmed via
                    # "Stream chunk: msg_type=AIMessageChunk" in the logs
                    # with zero assistant_delta events ever emitted).
                    continue

                if not content:
                    # Tool-call-only AI chunks (the model deciding to call a
                    # tool) have no text content — nothing to stream here.
                    continue

                stringified = _stringify_content(content)

                # A token (or small batch of tokens) of the assistant's
                # reply. The frontend appends these together rather than
                # replacing, so this actually streams in visibly instead
                # of arriving as one jump.
                final_assistant_content += stringified
                yield {
                    "event": "message",
                    "data": json.dumps({"type": "assistant_delta", "content": stringified}),
                }

            if final_assistant_content:
                await repository.append_message(
                    db, req.thread_id, role="assistant", content=final_assistant_content
                )

            yield {"event": "message", "data": json.dumps({"type": "done"})}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent run failed")
            yield {"event": "message", "data": json.dumps({"type": "error", "content": str(exc)})}

    return EventSourceResponse(event_generator())
