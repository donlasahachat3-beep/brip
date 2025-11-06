"""FastAPI router for managing targets."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from data.database import get_session
from services.target_manager.models import Target


router = APIRouter(prefix="/targets", tags=["targets"])


class TargetCreateRequest(BaseModel):
    id: str = Field(..., description="Target identifier")
    name: str
    url: str
    tags: list[str] = Field(default_factory=list)
    credentials: dict | None = None
    notes: str | None = None


class TargetResponse(BaseModel):
    id: str
    name: str
    url: str
    tags: list[str] | None
    notes: str | None


@router.post("", response_model=TargetResponse)
async def create_target(payload: TargetCreateRequest, session: AsyncSession = Depends(get_session)) -> TargetResponse:
    result = await session.execute(select(Target).where(Target.id == payload.id))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Target already exists")
    target = Target(
        id=payload.id,
        name=payload.name,
        url=payload.url,
        tags=",".join(payload.tags),
        credentials=str(payload.credentials) if payload.credentials else None,
        notes=payload.notes,
    )
    session.add(target)
    await session.commit()
    return TargetResponse(id=target.id, name=target.name, url=target.url, tags=payload.tags, notes=target.notes)


@router.get("", response_model=list[TargetResponse])
async def list_targets(session: AsyncSession = Depends(get_session)) -> list[TargetResponse]:
    result = await session.execute(select(Target))
    targets = []
    for row in result.scalars():
        tags = row.tags.split(",") if row.tags else []
        targets.append(TargetResponse(id=row.id, name=row.name, url=row.url, tags=tags, notes=row.notes))
    return targets
