"""Celery task definitions for orchestrating agent work."""

from __future__ import annotations

import httpx

from core.orchestrator.celery_app import app
from config.loader import get_settings


@app.task(name="orchestrator.dispatch_task")
def dispatch_task(agent_type: str, payload: dict) -> dict:
    """Dispatch a task to the agent manager for routing to agents."""

    settings = get_settings()
    manager_url = payload.get("manager_url") or "http://localhost:8100"  # placeholder default
    endpoint = f"{manager_url}/agents"

    try:
        response = httpx.post(endpoint, json={
            "agent_id": payload.get("agent_id", ""),
            "agent_type": agent_type,
            "capabilities": payload.get("capabilities", []),
            "state": payload.get("state", "queued"),
        }, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        return {"error": str(exc)}
