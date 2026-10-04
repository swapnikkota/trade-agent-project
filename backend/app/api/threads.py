"""
Thread (conversation) management endpoints. The React frontend (Phase 4)
uses these to drive a sidebar of past conversations: create a new thread,
list a user's threads, fetch a thread's message history.
"""
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import repository
from app.db.session import get_db
from app.rate_limit import limiter

router = APIRouter(prefix="/threads", tags=["threads"])

# Phase 6: looser cap than /chat since these are cheap DB reads/writes,
# not model calls — just enough to stop a runaway frontend loop (e.g. a
# buggy polling effect) from hammering Postgres.
_THREADS_RATE_LIMIT = "60/minute"


class CreateThreadRequest(BaseModel):
    user_id: str
    title: str | None = None


class RenameThreadRequest(BaseModel):
    title: str


class ThreadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    created_at: datetime


@router.post("", response_model=ThreadOut)
@limiter.limit(_THREADS_RATE_LIMIT)
async def create_thread(request: Request, req: CreateThreadRequest, db: AsyncSession = Depends(get_db)):
    thread = await repository.create_thread(db, user_id=req.user_id, title=req.title)
    return thread


@router.get("", response_model=list[ThreadOut])
@limiter.limit(_THREADS_RATE_LIMIT)
async def list_threads(request: Request, user_id: str, db: AsyncSession = Depends(get_db)):
    return await repository.list_threads(db, user_id=user_id)


@router.patch("/{thread_id}", response_model=ThreadOut)
@limiter.limit(_THREADS_RATE_LIMIT)
async def rename_thread(
    request: Request, thread_id: uuid.UUID, req: RenameThreadRequest, db: AsyncSession = Depends(get_db)
):
    thread = await repository.rename_thread(db, thread_id, req.title)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@router.delete("/{thread_id}", status_code=204)
@limiter.limit(_THREADS_RATE_LIMIT)
async def delete_thread(request: Request, thread_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    deleted = await repository.delete_thread(db, thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Thread not found")
    return None


@router.get("/{thread_id}/messages", response_model=list[MessageOut])
@limiter.limit(_THREADS_RATE_LIMIT)
async def get_thread_messages(request: Request, thread_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    thread = await repository.get_thread(db, thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return await repository.get_all_messages(db, thread_id)
