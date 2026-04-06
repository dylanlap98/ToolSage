"""
CallLogger — records every tool invocation to a per-tool JSON log file.

Each tool gets its own file: logs/log_{tool_name}.json

File format:
{
  "category_registry": {
    "<category_name>": "<one-sentence description of what this category represents>"
  },
  "entries": [ ... call records ... ]
}

The category_registry is populated and maintained by the Scorer as calls are classified.
It ensures the classifier is consistent across runs — new categories are only added when
no existing category fits.
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


class CallLogger:
    def __init__(self, log_dir: str | Path = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, tool_name: str) -> Path:
        return self.log_dir / f"log_{tool_name}.json"

    @staticmethod
    def _read(path: Path) -> tuple[dict, list]:
        """Return (category_registry, entries). Handles missing files and old array format."""
        if not path.exists():
            return {}, []
        content = path.read_text()
        if not content.strip():
            return {}, []
        data = json.loads(content)
        if isinstance(data, list):
            # Migrate old format (bare array) to new format
            return {}, data
        return data.get("category_registry", {}), data.get("entries", [])

    @staticmethod
    def _write(path: Path, registry: dict, entries: list) -> None:
        path.write_text(json.dumps({"category_registry": registry, "entries": entries}, indent=2))

    def write(
        self,
        tool_name: str,
        inputs: dict,
        output: str | None,
        duration_ms: float,
        error: str | None = None,
        task: str | None = None,
    ) -> None:
        entry = {
            "call_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "task": task,
            "inputs": inputs,
            "output": output,
            "duration_ms": round(duration_ms, 2),
            "error": error,
        }

        path = self._path(tool_name)
        with self._lock:
            registry, entries = self._read(path)
            entries.append(entry)
            self._write(path, registry, entries)
