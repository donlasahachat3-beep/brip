"""Celery application configuration."""

from __future__ import annotations

from celery import Celery

from config.loader import get_settings


def create_celery_app() -> Celery:
    settings = get_settings()
    celery = Celery(
        "dlnk_orchestrator",
        broker=settings.rabbitmq_url,
        backend=settings.redis_url,
    )
    celery.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
    )
    return celery


app = create_celery_app()
