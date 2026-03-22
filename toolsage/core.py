import inspect
from functools import wraps
from toolsage.manifest import ToolManifest

class ToolSage:
    def __init__(self):
        self.registry = {}

    def tool(self, manifest_path: str):
        def decorator(func):
            manifest = ToolManifest(manifest_path)

            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    return await func(*args, **kwargs)
            else:
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)

            wrapper.__doc__ = manifest.inject(func.__doc__ or "")
            self.registry[func.__name__] = {
                "manifest": manifest,
                "func": wrapper
            }
            return wrapper
        return decorator