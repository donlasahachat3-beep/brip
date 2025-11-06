"""FastAPI application exposing agent manager endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.loader import get_settings
from core.agent_manager.manager import AgentRecord, AgentRegistry


class AgentRegistrationRequest(BaseModel):
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Categorical agent type")
    capabilities: List[str] = Field(default_factory=list)
    state: str = Field(default="idle")


class AgentHeartbeatRequest(BaseModel):
    state: str = Field(..., description="Current agent state")


class AgentStatusResponse(BaseModel):
    agent_id: str
    agent_type: str
    capabilities: List[str]
    last_seen: datetime
    state: str


def get_registry() -> AgentRegistry:
    settings = get_settings()
    return AgentRegistry(redis_url=settings.redis_url)


app = FastAPI(title="Agent Manager", version="0.1.0")


@app.post("/agents", response_model=AgentStatusResponse)
async def register_agent(payload: AgentRegistrationRequest, registry: AgentRegistry = Depends(get_registry)) -> AgentStatusResponse:
    record = await registry.register(payload.agent_id, payload.agent_type, payload.model_dump())
    return AgentStatusResponse(**record.__dict__)


@app.post("/agents/{agent_id}/heartbeat")
async def heartbeat(agent_id: str, payload: AgentHeartbeatRequest, registry: AgentRegistry = Depends(get_registry)) -> dict:
    await registry.heartbeat(agent_id, payload.state)
    return {"status": "ok"}


@app.get("/agents", response_model=list[AgentStatusResponse])
async def list_agents(registry: AgentRegistry = Depends(get_registry)) -> list[AgentStatusResponse]:
    records = await registry.list_agents()
    return [AgentStatusResponse(**record.__dict__) for record in records]


@app.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str, registry: AgentRegistry = Depends(get_registry)) -> dict:
    await registry.unregister(agent_id)
    return {"status": "removed", "agent_id": agent_id}
