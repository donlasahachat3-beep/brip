"""Redis Pub/Sub communication layer."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Optional

import redis.asyncio as aioredis

from config.loader import get_settings


class RedisPubSub:
    def __init__(self, channel: str, redis_url: Optional[str] = None) -> None:
        settings = get_settings()
        self.redis_url = redis_url or settings.redis_url
        self.channel = channel
        self._redis = aioredis.from_url(self.redis_url, decode_responses=True)

    async def publish(self, message: str) -> None:
        await self._redis.publish(self.channel, message)

    async def subscribe(self) -> AsyncIterator[str]:
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self.channel)

        async for message in pubsub.listen():
            if message["type"] == "message":
                yield message["data"]

    async def close(self) -> None:
        await self._redis.close()
