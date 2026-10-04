"""
Async SQLAlchemy engine + session factory.

Phase 2 creates tables directly via metadata.create_all on startup (fine
for a project this size). A real migration tool (Alembic) is worth
introducing once the schema starts changing after data already exists —
not needed yet.
"""
import ssl
from collections.abc import AsyncGenerator

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
