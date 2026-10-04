"""
CRUD operations for users/threads/messages. Kept as plain functions
(no repository class) — this is a small enough app that a thin function
layer over SQLAlchemy is easier to follow than a repository abstraction.
"""
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Message, Thread, User


async def get_or_create_user(db: AsyncSession, user_id: str) -> User:
    user = await db.get(User, user_id)
    if user is None:
        user = User(id=user_id)
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user


async def create_thread(db: AsyncSession, user_id: str, title: str | None = None) -> Thread:
    await get_or_create_user(db, user_id)
    thread = Thread(user_id=user_id, title=title)
    db.add(thread)
    await db.commit()
    await db.refresh(thread)
    return thread


async def list_threads(db: AsyncSession, user_id: str) -> list[Thread]:
    result = await db.execute(
        select(Thread).where(Thread.user_id == user_id).order_by(Thread.updated_at.desc())
    )
    return list(result.scalars().all())


async def get_thread(db: AsyncSession, thread_id: uuid.UUID) -> Thread | None:
    return await db.get(Thread, thread_id)


async def rename_thread(db: AsyncSession, thread_id: uuid.UUID, title: str) -> Thread | None:
    thread = await db.get(Thread, thread_id)
    if thread is None:
        return None
    thread.title = title
    await db.commit()
    await db.refresh(thread)
    return thread


async def delete_thread(db: AsyncSession, thread_id: uuid.UUID) -> bool:
    """
    Deletes a thread and its messages. Messages are deleted explicitly
    (rather than relying on the ORM relationship cascade) to avoid an
    async lazy-load of thread.messages during the delete.
    """
    thread = await db.get(Thread, thread_id)
    if thread is None:
        return False
    await db.execute(delete(Message).where(Message.thread_id == thread_id))
    await db.delete(thread)
    await db.commit()
    return True


async def get_recent_messages(db: AsyncSession, thread_id: uuid.UUID, limit: int) -> list[Message]:
    """
    Returns the most recent `limit` messages for a thread, in chronological
    (oldest-first) order — ready to feed straight into the agent as context.
    """
    result = await db.execute(
        select(Message)
        .where(Message.thread_id == thread_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = list(result.scalars().all())
    messages.reverse()
    return messages


async def get_all_messages(db: AsyncSession, thread_id: uuid.UUID) -> list[Message]:
    result = await db.execute(
        select(Message).where(Message.thread_id == thread_id).order_by(Message.created_at.asc())
    )
    return list(result.scalars().all())


async def append_message(db: AsyncSession, thread_id: uuid.UUID, role: str, content: str) -> Message:
    message = Message(thread_id=thread_id, role=role, content=content)
    db.add(message)

    thread = await db.get(Thread, thread_id)
    if thread is not None:
        # Touch updated_at so thread lists sort by most-recently-active.
        from sqlalchemy import func

        thread.updated_at = func.now()

    await db.commit()
    await db.refresh(message)
    return message
