import inspect
import time
from functools import wraps
from pathlib import Path

from toolsage.logger import CallLogger
from toolsage.manifest import ToolManifest
from toolsage.scorer import Scorer


class ToolSage:
    def __init__(self, log_dir: str | Path = "logs"):
        self.registry = {}
        self._logger = CallLogger(log_dir)
        self._task: str | None = None

    def set_task(self, task: str) -> None:
        """Optional. Set the agent's current task/goal so it's recorded in every
        log entry. Useful context for the scorer — without it, task logs as null
        and scoring falls back to evaluating tool calls in isolation."""
        self._task = task

    def clear_task(self) -> None:
        self._task = None

    def score(self, llm=None) -> None:
        """Run LLM-as-judge over all logged tool calls. Each entry gets
        output_quality and manifest_adherence scored independently. Only
        unscored entries are evaluated; results are written back into the log."""
        scorer = Scorer(llm)
        for tool_name, reg in self.registry.items():
            log_path = self._logger._path(tool_name)
            if not log_path.exists():
                continue
            count = scorer.score_log(log_path, reg["manifest"].content)
            print(f"Scored {count} call(s) for '{tool_name}' → {log_path}")

    def tool(self, manifest_path: str):
        def decorator(func):
            manifest = ToolManifest(manifest_path)
            sig = inspect.signature(func)

            def _capture_inputs(*args, **kwargs) -> dict:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return dict(bound.arguments)

            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    inputs = _capture_inputs(*args, **kwargs)
                    start = time.monotonic()
                    try:
                        result = await func(*args, **kwargs)
                        self._logger.write(func.__name__, inputs, str(result), (time.monotonic() - start) * 1000, task=self._task)
                        return result
                    except Exception as e:
                        self._logger.write(func.__name__, inputs, None, (time.monotonic() - start) * 1000, error=str(e), task=self._task)
                        raise
            else:
                @wraps(func)
                def wrapper(*args, **kwargs):
                    inputs = _capture_inputs(*args, **kwargs)
                    start = time.monotonic()
                    try:
                        result = func(*args, **kwargs)
                        self._logger.write(func.__name__, inputs, str(result), (time.monotonic() - start) * 1000, task=self._task)
                        return result
                    except Exception as e:
                        self._logger.write(func.__name__, inputs, None, (time.monotonic() - start) * 1000, error=str(e), task=self._task)
                        raise

            wrapper.__doc__ = manifest.inject(func.__doc__ or "")
            self.registry[func.__name__] = {
                "manifest": manifest,
                "func": wrapper
            }
            return wrapper
        return decorator
