"""
Async SQLAlchemy engine + session factory.

Phase 2 creates tables directly via metadata.create_all on startup (fine
for a project this size). A real migration tool (Alembic) is worth
introducing once the schema starts changing after data already exists —
not needed yet.
"""
import re
import ssl
from collections.abc import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.models import Base

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_db(settings: Settings) -> AsyncEngine:
    global _engine, _sessionmaker
    connect_args = {}
    if settings.postgres_ssl:
        # Neon (and most hosted Postgres) require SSL. SQLAlchemy's
        # asyncpg dialect needs an actual SSL context here rather than a
        # "sslmode=require" query param (see app/config.py's database_url
        # for why that's stripped out instead of passed through).
        connect_args["ssl"] = ssl.create_default_context()
    _engine = create_async_engine(
        settings.database_url, echo=False, pool_pre_ping=True, connect_args=connect_args
    )
    if settings.postgres_schema:
        # Puts our schema first on the connection's search_path, so the
        # unqualified table names in app/db/models.py (and create_all's
        # CREATE TABLE statements) resolve there instead of "public" —
        # needed when this database is shared with other apps via
        # separate schemas (see app/config.py's postgres_schema).
        #
        # NOT done via connect_args={"server_settings": {...}} — confirmed
        # against the live trades-api deployment that Neon's connection
        # proxy silently drops the "search_path" startup parameter (every
        # connection came back with the default "$user", public regardless
        # of server_settings). So instead we run "SET search_path" as a
        # real query on every new pooled connection, via SQLAlchemy's
        # sync "connect" pool event (works for the asyncpg dialect too —
        # SQLAlchemy adapts the DBAPI connection to a sync-looking
        # interface here). Schema name is restricted to a plain identifier
        # since it's interpolated directly into SQL.
        schema = settings.postgres_schema
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
            raise ValueError(f"postgres_schema={schema!r} is not a valid unquoted Postgres identifier")

        @event.listens_for(_engine.sync_engine, "connect")
        def _set_search_path(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute(f"SET search_path TO {schema}")
            cursor.close()

    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def create_all_tables() -> None:
    assert _engine is not None, "init_db() must be called before create_all_tables()"
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db() -> None:
    if _engine is not None:
        await _engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    assert _sessionmaker is not None, "init_db() must be called before get_db()"
    async with _sessionmaker() as session:
        yield session
