"""Structured logging utilities for the monitoring platform.

This module exposes pre-configured JSON loggers for the various monitoring
concerns described in the project README. The implementation focuses on:

* JSON-formatted output with a consistent schema
* Optional contextual metadata that is automatically injected into all
  log records produced within a logical scope
* Automatic log rotation with sensible defaults
* Convenience helpers for audit, security, performance, and general purpose
  structured logging needs

The implementation builds on Python's standard :mod:`logging` module to avoid
additional runtime dependencies and remains thread-safe thanks to the
thread-local storage used for contextual data.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


__all__ = [
    "StructuredLogger",
    "AuditLogger",
    "SecurityLogger",
    "PerformanceLogger",
    "get_logger",
    "logging_context",
]


LOG_DIR_ENV_VAR = "DLNK_MONITORING_LOG_DIR"
DEFAULT_LOG_DIR = Path("logs")


class _ContextStore(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[dict[str, Any]] = []

    def push(self, context: dict[str, Any]) -> None:
        self.stack.append(context)

    def pop(self) -> None:
        if self.stack:
            self.stack.pop()

    def flatten(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for ctx in self.stack:
            merged.update(ctx)
        return merged


_context_store = _ContextStore()


class JsonLogFormatter(logging.Formatter):
    """Format log records as JSON with consistent schema."""

    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03dZ"

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:  # noqa: N802 - inherited signature
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3]

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - stdlib interface
        base_payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            base_payload["exc_info"] = self.formatException(record.exc_info)

        # Merge context metadata and arbitrary extra fields stored on the record
        metadata = _context_store.flatten()
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in logging.LogRecord.__dict__:
                continue
            metadata.setdefault(key, value)

        if metadata:
            base_payload["metadata"] = metadata

        return json.dumps(base_payload, default=str, ensure_ascii=False)


def _resolve_log_dir() -> Path:
    raw = os.getenv(LOG_DIR_ENV_VAR)
    path = Path(raw) if raw else DEFAULT_LOG_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_handler(log_name: str) -> logging.Handler:
    log_dir = _resolve_log_dir()
    log_file = log_dir / f"{log_name}.log"
    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    handler.setFormatter(JsonLogFormatter())
    return handler


def _configure_logger(logger_name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(level)
        logger.propagate = False
        logger.addHandler(_build_handler(logger_name))
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(JsonLogFormatter())
        logger.addHandler(stream_handler)
    return logger


StructuredLogger = _configure_logger("structured")
AuditLogger = _configure_logger("audit")
SecurityLogger = _configure_logger("security")
PerformanceLogger = _configure_logger("performance")


def get_logger(name: str) -> logging.Logger:
    """Return a structured logger with the requested name.

    The logger will automatically emit JSON-formatted output and rotate files.
    """

    return _configure_logger(name)


@contextmanager
def logging_context(**metadata: Any) -> Iterator[None]:
    """Context manager that injects metadata into all log records.

    Example
    -------
    >>> with logging_context(agent_id="agent-123"):
    ...     AuditLogger.info("Agent started")
    """

    _context_store.push(metadata)
    try:
        yield
    finally:
        _context_store.pop()


def set_global_log_level(level: int) -> None:
    """Update the log level for all configured structured loggers."""

    for logger_name in {"structured", "audit", "security", "performance"}:
        logging.getLogger(logger_name).setLevel(level)


def reset_loggers() -> None:
    """Remove handlers for all known loggers (useful for testing)."""

    for logger_name in {"structured", "audit", "security", "performance"}:
        logger = logging.getLogger(logger_name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

