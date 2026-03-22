from pathlib import Path

class ToolManifest:
    def __init__(self, path: str):
        self.path = Path(path)
        self.content = self._load()

    def _load(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.path}")
        return self.path.read_text()

    def inject(self, base_description: str) -> str:
        return f"{base_description}\n\n---\n{self.content}"