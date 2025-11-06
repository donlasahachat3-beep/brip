"""FastAPI-based monitoring API exposing system metrics and controls."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.monitoring import MonitoringCore
from infrastructure.monitoring.metrics import global_metrics
from infrastructure.monitoring.structured_logger import (
    AuditLogger,
    PerformanceLogger,
    SecurityLogger,
    StructuredLogger,
    logging_context,
)


LOG_COMPONENTS = {"structured", "audit", "security", "performance"}


def _read_log(component: str, line_limit: int) -> List[str]:
    if component not in LOG_COMPONENTS:
        raise HTTPException(status_code=404, detail="Unknown log component")

    log_dir = Path(os.getenv("DLNK_MONITORING_LOG_DIR", "logs"))
    log_file = log_dir / f"{component}.log"
    if not log_file.exists():
        return []

    with log_file.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
        return lines[-line_limit:]


class AgentStatusStore:
    """Redis-backed store for agent status and metrics."""

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._client = redis.from_url(redis_url) if redis_url else None

    def available(self) -> bool:
        if not self._client:
            return False
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            SecurityLogger.warning("Redis ping failed", extra={"redis_configured": True})
            return False

    def list_agents(self) -> List[Dict[str, Any]]:
        if not self._client:
            return []
        agents: List[Dict[str, Any]] = []
        try:
            for key in self._client.scan_iter(match="agent:*"):
                data = self._client.hgetall(key)
                agents.append({k.decode("utf-8"): v.decode("utf-8") for k, v in data.items()})
        except redis.RedisError as exc:
            SecurityLogger.exception("Failed to list agents", extra={"error": str(exc)})
        return agents


@lru_cache(maxsize=1)
def _monitoring_core() -> MonitoringCore:
    return MonitoringCore()


@lru_cache(maxsize=1)
def _agent_store() -> AgentStatusStore:
    redis_url = os.getenv("MONITORING_REDIS_URL")
    return AgentStatusStore(redis_url=redis_url)


app = FastAPI(title="dLNK Monitoring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    StructuredLogger.info("Monitoring API startup")


@app.middleware("http")
async def request_metrics_middleware(request, call_next):  # type: ignore[annotations-unchecked]
    monitoring = get_monitoring_core()
    path = request.url.path
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    monitoring.record_request(endpoint=path, status=str(response.status_code), duration_s=duration)
    PerformanceLogger.info(
        "HTTP request",
        extra={"path": path, "status": response.status_code, "duration_seconds": duration},
    )
    return response


def get_monitoring_core() -> MonitoringCore:
    return _monitoring_core()


def get_agent_store() -> AgentStatusStore:
    return _agent_store()


@app.get("/")
async def root_endpoint() -> Dict[str, Any]:
    return {
        "service": app.title,
        "version": app.version,
        "documentation": {
            "openapi": "/openapi.json",
            "swagger_ui": "/docs",
            "redoc": "/redoc",
        },
        "available_endpoints": [
            "/health",
            "/system",
            "/status",
            "/metrics",
            "/agents",
            "/logs/{component}",
            "/trigger-self-healing",
            "/attacks/{attack_type}/record",
        ],
    }


@app.get("/favicon.ico")
async def favicon_endpoint() -> Response:
    return Response(status_code=204)


@app.get("/health")
async def health_endpoint(monitoring: MonitoringCore = Depends(get_monitoring_core)) -> Dict[str, Any]:
    checks = monitoring.health_checks()
    payload = {
        "system": monitoring.system_metrics(),
        "health": {name: check.__dict__ for name, check in checks.items()},
    }
    return payload


@app.get("/system")
async def system_endpoint(monitoring: MonitoringCore = Depends(get_monitoring_core)) -> Dict[str, Any]:
    return monitoring.system_metrics()


@app.get("/status")
async def status_endpoint(monitoring: MonitoringCore = Depends(get_monitoring_core)) -> Dict[str, Any]:
    health = monitoring.health_checks()
    return {
        "hostname": monitoring.hostname,
        "uptime_seconds": monitoring.uptime_seconds(),
        "overall_status": health["overall"].healthy,
        "application_metrics": monitoring.application_metrics(),
    }


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    registry = global_metrics()
    return Response(content=registry.expose(), media_type="text/plain; version=0.0.4")


@app.get("/agents")
async def agents_endpoint(store: AgentStatusStore = Depends(get_agent_store)) -> Dict[str, Any]:
    agents = store.list_agents()
    return {
        "available": store.available(),
        "agents": agents,
        "count": len(agents),
    }


@app.get("/logs/{component}")
async def logs_endpoint(component: str, lines: int = 200) -> Dict[str, Any]:
    lines = max(0, min(lines, 1000))
    entries = _read_log(component, line_limit=lines)
    return {"component": component, "lines": entries}


@app.post("/trigger-self-healing")
async def trigger_self_healing(reason: Optional[str] = None) -> Dict[str, Any]:
    with logging_context(action="trigger-self-healing"):
        AuditLogger.warning("Self-healing triggered", extra={"reason": reason})
    return {"status": "triggered", "reason": reason}


@app.post("/attacks/{attack_type}/record")
async def record_attack(attack_type: str, result: str, monitoring: MonitoringCore = Depends(get_monitoring_core)) -> Dict[str, Any]:
    monitoring.record_attack_event(attack_type=attack_type, result=result)
    SecurityLogger.info(
        "Attack recorded",
        extra={"attack_type": attack_type, "result": result},
    )
    return {"status": "recorded"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # type: ignore[annotations-unchecked]
    SecurityLogger.exception("Unhandled API error", extra={"path": str(request.url)})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def run() -> None:
    import uvicorn

    uvicorn.run("app.monitoring_api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)

