"""Auto-detection service stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import httpx
from bs4 import BeautifulSoup


@dataclass
class DetectionResult:
    target: str
    technologies: List[str]
    notes: str | None = None


class AutoDetectionService:
    async def analyze(self, url: str) -> DetectionResult:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        techs = []
        if soup.find("meta", attrs={"name": "generator"}):
            techs.append("cms")
        if "wp-content" in response.text:
            techs.append("wordpress")
        return DetectionResult(target=url, technologies=techs)
