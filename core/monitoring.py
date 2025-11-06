"""Core monitoring utilities that aggregate metrics and health checks."""

from __future__ import annotations

import functools
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import psutil

from infrastructure.monitoring.metrics import global_metrics
from infrastructure.monitoring.structured_logger import PerformanceLogger


HealthCheckFunc = Callable[[], tuple[bool, Optional[str]]]


@dataclass
class HealthCheckResult:
    name: str
    healthy: bool
    details: Optional[str] = None


def _default_db_check() -> tuple[bool, Optional[str]]:
    return False, "database check not configured"


def _default_redis_check() -> tuple[bool, Optional[str]]:
    return False, "redis check not configured"


def _default_disk_threshold_check(threshold: float = 0.9) -> tuple[bool, Optional[str]]:
    usage = psutil.disk_usage("/")
    ratio = usage.used / usage.total
    return ratio < threshold, f"disk usage at {ratio:.2%}"


def _default_memory_threshold_check(threshold: float = 0.9) -> tuple[bool, Optional[str]]:
    usage = psutil.virtual_memory()
    ratio = usage.percent / 100.0
    return ratio < threshold, f"memory usage at {ratio:.2%}"


class MonitoringCore:
    """High-level orchestrator for system, application, and health monitoring."""

    def __init__(
        self,
        *,
        db_check: Optional[HealthCheckFunc] = None,
        redis_check: Optional[HealthCheckFunc] = None,
        disk_threshold_check: Optional[HealthCheckFunc] = None,
        memory_threshold_check: Optional[HealthCheckFunc] = None,
        hostname: Optional[str] = None,
    ) -> None:
        self.hostname = hostname or socket.gethostname()
        self._startup_ts = time.time()
        self._db_check = db_check or _default_db_check
        self._redis_check = redis_check or _default_redis_check
        self._disk_check = disk_threshold_check or _default_disk_threshold_check
        self._memory_check = memory_threshold_check or _default_memory_threshold_check

        self._metrics = global_metrics()
        self._lock = threading.RLock()
        self._register_metrics()
        self._request_total = 0
        self._attack_total = 0

    # ------------------------------------------------------------------
    # Metric recording helpers
    # ------------------------------------------------------------------
    def record_request(self, endpoint: str, status: str, duration_s: float) -> None:
        with self._lock:
            self._request_counter.labels(endpoint=endpoint, status=status).inc()
            self._request_duration.labels(endpoint=endpoint).observe(duration_s)
            self._request_total += 1

    def record_attack_event(self, attack_type: str, result: str) -> None:
        with self._lock:
            self._attack_counter.labels(type=attack_type, result=result).inc()
            self._attack_total += 1

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def system_metrics(self) -> Dict[str, Any]:
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        net_io = psutil.net_io_counters()
        swap = psutil.swap_memory()
        load_avg = os.getloadavg() if hasattr(os, "getloadavg") else None

        metrics = {
            "hostname": self.hostname,
            "cpu_percent": cpu_percent,
            "load_average": load_avg,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free,
            },
            "swap": {
                "total": swap.total,
                "used": swap.used,
                "percent": swap.percent,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
            },
            "uptime_seconds": self.uptime_seconds(),
        }

        self._uptime_gauge.set(metrics["uptime_seconds"])
        return metrics

    def application_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "requests_total": self._request_total,
                "attacks_total": self._attack_total,
                "uptime_seconds": self.uptime_seconds(),
            }

    def health_checks(self) -> Dict[str, HealthCheckResult]:
        checks = {
            "database": self._evaluate("database", self._db_check),
            "redis": self._evaluate("redis", self._redis_check),
            "disk": self._evaluate("disk", self._disk_check),
            "memory": self._evaluate("memory", self._memory_check),
        }
        checks["overall"] = HealthCheckResult(
            name="overall",
            healthy=all(check.healthy for check in checks.values()),
        )
        return checks

    def uptime_seconds(self) -> float:
        return time.time() - self._startup_ts

    def reset_metrics(self) -> None:
        with self._lock:
            self._metrics.reset()
            self._register_metrics()
            self._request_total = 0
            self._attack_total = 0

    def performance_monitor(self, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to measure execution time and emit metrics/logs."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            metric_name = name or func.__qualname__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                with self._performance_histogram.labels(function=metric_name).time():
                    result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                PerformanceLogger.info(
                    "Function execution",
                    extra={"function": metric_name, "duration_seconds": duration},
                )
                return result

            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _evaluate(name: str, check: HealthCheckFunc) -> HealthCheckResult:
        healthy, details = check()
        return HealthCheckResult(name=name, healthy=healthy, details=details)

    def _register_metrics(self) -> None:
        # Register core metrics with the shared registry
        self._request_counter = self._metrics.register_counter(
            "app_requests_total",
            "Total number of processed requests",
            labelnames=("endpoint", "status"),
        )
        self._attack_counter = self._metrics.register_counter(
            "attack_events_total",
            "Total number of tracked attack executions",
            labelnames=("type", "result"),
        )
        self._request_duration = self._metrics.register_summary(
            "request_duration_seconds",
            "Request handling duration",
            labelnames=("endpoint",),
        )
        self._performance_histogram = self._metrics.register_histogram(
            "monitored_function_seconds",
            "Function execution duration",
            labelnames=("function",),
        )
        self._uptime_gauge = self._metrics.register_gauge(
            "app_uptime_seconds",
            "Application uptime",
        )


__all__ = ["MonitoringCore", "HealthCheckResult"]

