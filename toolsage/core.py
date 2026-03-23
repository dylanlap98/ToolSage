import inspect
import time
from functools import wraps
from pathlib import Path

from toolsage.logger import CallLogger
from toolsage.manifest import ToolManifest


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
