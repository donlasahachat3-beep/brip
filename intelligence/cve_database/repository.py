"""CVE database abstraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class CVEDatabase:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("intelligence/cve_database/data.json")
        self._index: Dict[str, List[dict]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._index = {}
            return
        with self.path.open("r", encoding="utf-8") as fh:
            self._index = json.load(fh)

    def find_by_technology(self, technology: str) -> List[dict]:
        return self._index.get(technology, [])

    def all(self) -> Dict[str, List[dict]]:
        return self._index
