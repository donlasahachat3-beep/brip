"""Knowledge base repository backed by YAML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import yaml


class KnowledgeBase:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("intelligence/knowledge_base/data.yaml")
        self._data: Dict[str, List[str]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        with self.path.open("r", encoding="utf-8") as fh:
            self._data = yaml.safe_load(fh) or {}

    def lookup(self, technology: str) -> List[str]:
        return self._data.get(technology, [])

    def all_entries(self) -> Dict[str, List[str]]:
        return self._data
