# Trade Agent — Backend

FastAPI service hosting a deepagents agent that talks to your existing
trade MCP server, with Postgres-backed conversation history and Langfuse
Cloud tracing.

## Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:
- Model provider: defaults to `ollama` (free, local). Set `MODEL_PROVIDER=anthropic`
  and fill in `ANTHROPIC_API_KEY` to use Claude instead.
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (from your Langfuse Cloud project)
- `TRADE_MCP_URL` — already set to `http://localhost:9000/sse`
- Postgres settings — `POSTGRES_USER`/`POSTGRES_PASSWORD` for your local instance,
  `POSTGRES_DB=trade_agent`
- `MCP_SERVERS` (Phase 5, optional) — set this to connect to more than one MCP
  server at once instead of just the single trade server; see "Multiple MCP
  servers" below

### Create the database

```bash
createdb trade_agent
```

Tables (`users`, `threads`, `messages`) are created automatically on
startup — no separate migration step for now.

## Run

```bash
uvicorn app.main:app --reload --port 8080
```

## Try it

Conversation history now requires a **thread**. Create one first, then chat within it:

```bash
# 1. Create a thread
curl -X POST http://localhost:8080/threads \
  -H "Content-Type: application/json" \
  -d '{"user_id": "swapnik", "title": "AAPL trades"}'
# => {"id": "<thread_id>", "user_id": "swapnik", ...}

# 2. Chat within that thread (repeat with the same thread_id to continue the conversation)
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What trades happened today for AAPL?", "user_id": "swapnik", "thread_id": "<thread_id>"}'

# 3. List a user's threads
curl "http://localhost:8080/threads?user_id=swapnik"

# 4. Fetch full history for a thread
curl "http://localhost:8080/threads/<thread_id>/messages"
```

Each `/chat` call:
- loads the last `HISTORY_WINDOW_SIZE` messages from that thread as context
- appends the new user message + agent's final reply to the thread afterward
- tags the Langfuse trace with `user_id` and `session_id = thread_id`, so traces
  are filterable by conversation in Langfuse Cloud

## Multiple MCP servers (Phase 5)

By default the agent connects to just the trade MCP server (`TRADE_MCP_URL`).
To add more servers without touching code, set `MCP_SERVERS` in `.env` as a
JSON array — it takes over from `TRADE_MCP_*` entirely when set:

```
MCP_SERVERS=[{"name":"trade","transport":"sse","url":"http://localhost:9000/sse"},{"name":"market_data","transport":"sse","url":"http://localhost:9001/sse"}]
```

A small mock second server is included at `mock_servers/market_data_server.py`
— it's not a real data source, just a one-tool (`get_quote`) server for
proving the multi-server wiring actually works before you build or point at
a real second MCP server:

```bash
python mock_servers/market_data_server.py   # runs on port 9001
```

Restart the backend with `MCP_SERVERS` set as above, and check the startup
log line `Loaded N MCP tool(s): [...]` — it should list tools from both
servers (`get_trades`, `get_trade_by_id`, `create_trade`, `get_quote`).

## Auth & rate limiting (Phase 6)

**Auth.** Off by default (local dev). To turn it on, set `API_KEY` in
`.env` to a random secret. Once set, every request except `/health` (and
CORS preflight `OPTIONS`) must send it back via the `X-API-Key` header or
get a `401`. This is a single shared secret, not per-user accounts —
proportionate for a personal project used by one person, not a substitute
for real auth if this is ever exposed to more than one trusted person.

If you turn this on, the frontend needs to send the same value: set
`VITE_API_KEY` in `frontend/.env` to match `API_KEY` exactly, or every
request from the UI will 401.

```bash
curl http://localhost:8080/threads?user_id=swapnik \
  -H "X-API-Key: <your API_KEY>"
```

**Rate limiting.** Always on (doesn't depend on `API_KEY` being set) via
`slowapi`, keyed by the `X-API-Key` header when auth is configured
(so each key gets its own budget, stable across NATs/shared IPs) or by
remote address otherwise:

- `/chat` — `20/minute` (capped tighter: it's the expensive model-call endpoint)
- `/threads` endpoints (create/list/rename/delete/messages) — `60/minute`
  (cheap DB reads/writes, just enough to stop a runaway loop)

A request over the limit gets a `429` from slowapi's default handler.
These caps are hardcoded in `app/rate_limit.py` / the `@limiter.limit(...)`
decorators for now rather than env-configurable — revisit if this ever
needs tuning without a code change.

## What's here

- `app/config.py` — env-driven settings, incl. Postgres DSN builder
- `app/db/models.py` — `User`, `Thread`, `Message` SQLAlchemy models
- `app/db/session.py` — async engine/session, `create_all_tables()`
- `app/db/repository.py` — CRUD functions (create/list/delete threads, append/fetch messages)
- `app/api/threads.py` — `/threads` REST endpoints for the frontend's conversation sidebar
- `app/mcp/registry.py` — MCP server connection config; builds from
  `settings.mcp_servers` (Phase 5, config-driven N servers) or falls back
  to the single legacy trade server config
- `mock_servers/market_data_server.py` — throwaway second MCP server used
  only to test Phase 5's multi-server wiring
- `app/agent/factory.py` — deepagents agent construction + Langfuse
  callback handler factory
- `app/rate_limit.py` — shared slowapi `Limiter` instance (Phase 6), used
  by both `app/main.py` and `app/api/threads.py`
- `app/main.py` — FastAPI app, API-key auth middleware, `/chat` SSE
  endpoint (history-aware, persists turns, rate-limited)

## Not yet in scope (later phases)

- Nested MCP-call spans / richer Langfuse instrumentation review (Phase 3 — largely
  already working, worth a closer pass)
- Deployment (Phase 6, still to scope — Docker? a specific host? staying local?)
