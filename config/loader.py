"""Configuration loader utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    app_name: str = "dlnk_v5"
    environment: str = "development"
    redis_url: str = "redis://localhost:6379/0"
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672//"
    postgres_dsn: str = "postgresql+asyncpg://dlnk:password@localhost:5432/dlnk_v5"
    vc_api_key: str | None = None

    @classmethod
    def from_yaml(cls, data: Dict[str, Any]) -> "Settings":
        payload = {
            "app_name": data.get("app", {}).get("name", cls().app_name),
            "environment": data.get("app", {}).get("environment", cls().environment),
            "redis_url": data.get("redis", {}).get("url", cls().redis_url),
            "rabbitmq_url": data.get("rabbitmq", {}).get("url", cls().rabbitmq_url),
            "postgres_dsn": data.get("postgresql", {}).get("dsn", cls().postgres_dsn),
        }
        return cls(**payload)


def _load_yaml_settings() -> Dict[str, Any]:
    settings_path = Path("config/settings.yaml")
    if not settings_path.exists():
        return {}
    with settings_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data = _load_yaml_settings()
    return Settings.from_yaml(data)
