"""
Centralized settings for the backend. Loaded once, imported everywhere.
"""
from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerConfig(BaseModel):
    """
    One entry in the MCP_SERVERS list (Phase 5). Each entry describes a
    single MCP server the agent should connect to; `name` becomes the key
    MultiServerMCPClient/registry.py use to namespace that server's tools.
    """

    name: str
    transport: Literal["sse", "streamable_http", "stdio"]
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"

    # Comma-separated list of allowed CORS origins, e.g.
    # "http://localhost:5173,https://trade-agent-frontend.onrender.com".
    # Defaults to the local Vite dev server origins.
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # Phase 6: shared-secret API key. When set, every request (except
    # /health) must send it back via the X-API-Key header or get a 401.
    # Left empty (the default), auth is OFF — fine for local-only dev, but
    # means anyone who can reach the backend can read/write any user_id's
    # threads. This is a single shared secret, not per-user accounts —
    # proportionate for a personal project, not a substitute for real auth
    # if this is ever exposed to more than one trusted person.
    api_key: str = ""

    # Model provider: "anthropic" (paid, cloud), "ollama" (free, local —
    # requires a reachable Ollama instance, won't work once the backend is
    # hosted somewhere that isn't this machine), or "groq" (free, cloud —
    # the default for the hosted deployment).
    model_provider: Literal["anthropic", "ollama", "groq"] = "ollama"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"

    # Groq: free, fast inference. The Llama model lineup (llama-3.3-70b-
    # versatile, llama-3.1-8b-instant) is no longer available on at least
    # some Groq accounts (confirmed live: both 404 "does not exist or you
    # do not have access to it" despite being listed in Groq's docs) —
    # Groq's free-tier lineup has shifted toward their OpenAI OSS models.
    # openai/gpt-oss-120b is the largest of the models this account has
    # access to that supports tool calling (checked via GET
    # /openai/v1/models and filtering for "tools" in supported_features —
    # do that again if this default stops working, rather than guessing
    # another model name).
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # Trade MCP server (Phase 1: hardcoded single entry). Still read as a
    # fallback when MCP_SERVERS (below) isn't set, so existing .env files
    # keep working unchanged.
    trade_mcp_transport: Literal["http", "stdio"] = "http"
    trade_mcp_url: Optional[str] = None
    trade_mcp_command: Optional[str] = None
    trade_mcp_args: Optional[str] = None

    # Phase 5: config-driven list of MCP servers, e.g. as a JSON array in
    # the MCP_SERVERS env var:
    #   MCP_SERVERS=[{"name":"trade","transport":"sse","url":"http://localhost:9000/sse"},
    #                {"name":"market_data","transport":"stdio","command":"python","args":["mock_servers/market_data_server.py"]}]
    # When this is non-empty it takes over from trade_mcp_* entirely —
    # adding a new MCP server becomes a config change, not a code change.
    # Left empty (the default), the app falls back to the single trade_mcp_*
    # config above.
    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Postgres (session/thread/message history)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "swapnikkota"
    postgres_password: str = ""
    postgres_db: str = "trade_agent"

    # Set this (as DATABASE_URL in the environment) to use a full connection
    # string instead of the individual POSTGRES_* fields above — the form
    # a hosted Postgres provider (Neon, etc.) actually gives you. When set,
    # this takes over entirely; POSTGRES_* above is only the local-dev
    # fallback. Accepts either "postgresql://" or "postgresql+asyncpg://".
    database_url_override: str = Field(default="", validation_alias="DATABASE_URL")

    # Whether the Postgres connection needs SSL (true for Neon and most
    # hosted providers, false for a local instance). Kept as its own flag
    # rather than parsed out of a "?sslmode=require" query string, because
    # SQLAlchemy's asyncpg dialect doesn't accept "sslmode" the way psql/
    # psycopg2 do — it needs an actual SSL context passed via connect_args
    # (see app/db/session.py), so any "sslmode=..." in a pasted connection
    # string is stripped in database_url below and this flag used instead.
    postgres_ssl: bool = False

    # Which Postgres schema this app's tables live in. Empty (the default)
    # means "whatever the connection's default search_path resolves to" —
    # normally "public", fine for a local/standalone database. Set this
    # when the database is shared with other apps via separate schemas
    # (e.g. one Neon project/database holding both this app's "trade_agent"
    # schema and the trades-api's "tradesdb" schema) — the unqualified
    # table names in app/db/models.py then resolve via this schema being
    # first on the connection's search_path (see app/db/session.py).
    postgres_schema: str = ""

    # How many most-recent messages to feed back into the agent as context.
    # Phase 2: simple sliding window. A later phase can swap this for
    # summarization once conversations get long.
    history_window_size: int = 20

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            url = self.database_url_override
            # Normalize the scheme to SQLAlchemy's asyncpg dialect form —
            # hosted providers commonly hand out "postgres://" or
            # "postgresql://".
            if url.startswith("postgres://"):
                url = "postgresql://" + url[len("postgres://") :]
            if url.startswith("postgresql://") and "+asyncpg" not in url:
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            # Strip sslmode/channel_binding query params (libpq-style —
            # asyncpg.connect() doesn't accept them as kwargs and SQLAlchemy
            # would otherwise forward them verbatim and blow up). SSL itself
            # is handled via postgres_ssl + connect_args in session.py.
            if "?" in url:
                base, _, query = url.partition("?")
                kept = [
                    kv
                    for kv in query.split("&")
                    if kv and not kv.split("=")[0] in ("sslmode", "channel_binding")
                ]
                url = base + ("?" + "&".join(kept) if kept else "")
            return url

        auth = self.postgres_user
        if self.postgres_password:
            auth += f":{self.postgres_password}"
        return (
            f"postgresql+asyncpg://{auth}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
