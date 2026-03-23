"""
CallLogger — records every tool invocation to a per-tool JSON log file.

Each tool gets its own file: logs/log_{tool_name}.json
Each file is a JSON array of call records, one object per invocation.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class CallLogger:
    def __init__(self, log_dir: str | Path = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, tool_name: str) -> Path:
        return self.log_dir / f"log_{tool_name}.json"

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
        content = path.read_text() if path.exists() else ""
        existing = json.loads(content) if content.strip() else []

        existing.append(entry)
        path.write_text(json.dumps(existing, indent=2))
