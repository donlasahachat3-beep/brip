"""Base agent abstractions for dLNK v5."""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class AgentState(str, Enum):
    """Lifecycle states shared by all agent implementations."""

    INITIALIZING = "initializing"
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class AgentContext:
    """Runtime context provided to agents by the manager/orchestrator."""

    agent_id: str
    config: Dict[str, Any]
    shared_state: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(abc.ABC):
    """Abstract base class describing the agent interface."""

    def __init__(self, context: AgentContext) -> None:
        self.context = context
        self._state: AgentState = AgentState.INITIALIZING
        self._state_lock = asyncio.Lock()

    @property
    def state(self) -> AgentState:
        return self._state

    async def set_state(self, state: AgentState) -> None:
        async with self._state_lock:
            self._state = state

    async def initialize(self) -> None:
        """Perform optional asynchronous initialization."""

    @abc.abstractmethod
    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary logic for the given payload."""

    async def stop(self) -> None:
        """Gracefully stop the agent."""
        await self.set_state(AgentState.STOPPING)

    async def get_status(self) -> Dict[str, Any]:
        """Return a serializable status report for health monitoring."""

        return {
            "agent_id": self.context.agent_id,
            "state": self.state,
            "metadata": await self._status_metadata(),
        }

    async def _status_metadata(self) -> Dict[str, Any]:
        """Hook for subclasses to expose additional status metadata."""

        return {}

    async def register_self(self) -> Dict[str, Any]:
        """Payload returned to the agent manager during registration."""

        return {
            "agent_id": self.context.agent_id,
            "capabilities": self.context.config.get("capabilities", []),
        }
