"""Agent manager core logic."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import redis.asyncio as aioredis


@dataclass
class AgentRecord:
    agent_id: str
    agent_type: str
    capabilities: List[str]
    last_seen: datetime
    state: str


class AgentRegistry:
    """Redis-backed agent registry."""

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._redis: Optional[aioredis.Redis] = None
        self._lock = asyncio.Lock()

    async def _connection(self) -> aioredis.Redis:
        if self._redis is None:
            async with self._lock:
                if self._redis is None:
                    self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def register(self, agent_id: str, agent_type: str, payload: Dict[str, any]) -> AgentRecord:
        conn = await self._connection()
        record = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "capabilities": payload.get("capabilities", []),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "state": payload.get("state", "idle"),
        }
        await conn.hset("agents", agent_id, json.dumps(record))
        return AgentRecord(
            agent_id=record["agent_id"],
            agent_type=record["agent_type"],
            capabilities=record["capabilities"],
            last_seen=datetime.fromisoformat(record["last_seen"]),
            state=record["state"],
        )

    async def heartbeat(self, agent_id: str, state: str) -> None:
        conn = await self._connection()
        data = await conn.hget("agents", agent_id)
        if not data:
            return
        record = json.loads(data)
        record["state"] = state
        record["last_seen"] = datetime.now(timezone.utc).isoformat()
        await conn.hset("agents", agent_id, json.dumps(record))

    async def list_agents(self) -> List[AgentRecord]:
        conn = await self._connection()
        results = []
        entries = await conn.hgetall("agents")
        for raw in entries.values():
            data = json.loads(raw)
            results.append(
                AgentRecord(
                    agent_id=data["agent_id"],
                    agent_type=data["agent_type"],
                    capabilities=data.get("capabilities", []),
                    last_seen=datetime.fromisoformat(data["last_seen"]),
                    state=data.get("state", "idle"),
                )
            )
        return results

    async def unregister(self, agent_id: str) -> None:
        conn = await self._connection()
        await conn.hdel("agents", agent_id)


__all__ = ["AgentRegistry", "AgentRecord"]
