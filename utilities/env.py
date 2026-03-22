"""
populate_env — loads secrets into os.environ at runtime.

Reads a manifest file (secrets-manifest.env) that maps env var names to
JSONPath-style selectors, then resolves those paths against a secrets JSON file.

Manifest format:
    ANTHROPIC_API_KEY: $.anthropic.ANTHROPIC_API_KEY
    SOME_OTHER_KEY: $.service.nested.value

JSONPath format: $.section.subsection.key  (simple dotted path, no array support needed)
"""

import json
import os
from pathlib import Path


_REPO_ROOT = Path(__file__).parent.parent  # utilities/ -> repo root
_DEFAULT_MANIFEST = _REPO_ROOT / "secrets-manifest.env"
_DEFAULT_SECRETS = _REPO_ROOT / "secrets" / "secrets.json"


def _resolve_path(data: dict, path: str) -> str:
    """Resolve a $.a.b.c path against a nested dict."""
    if not path.startswith("$."):
        raise ValueError(f"Invalid path format '{path}' — must start with '$.'")
    parts = path[2:].split(".")
    node = data
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Path '{path}' not found in secrets (missing key: '{part}')")
        node = node[part]
    if not isinstance(node, str):
        raise TypeError(f"Path '{path}' resolved to {type(node).__name__}, expected str")
    return node


def populate_env(
    manifest_path: str | Path = _DEFAULT_MANIFEST,
    secrets_path: str | Path = _DEFAULT_SECRETS,
    overwrite: bool = False,
) -> dict[str, str]:
    """
    Read the manifest and load mapped secrets into os.environ.

    Returns a dict of {ENV_VAR: value} for the vars that were set,
    so callers can log or inspect what was populated (without logging values).

    Args:
        manifest_path: Path to the .env manifest file. Defaults to
                       toolsage/secrets-manifest.env.
        secrets_path:  Path to the secrets JSON file. Defaults to
                       toolsage/secrets/secrets.json.
        overwrite:     If False (default), skip vars already set in the environment.
    """
    manifest_path = Path(manifest_path)
    secrets_path = Path(secrets_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Secrets manifest not found: {manifest_path}")
    if not secrets_path.exists():
        raise FileNotFoundError(f"Secrets file not found: {secrets_path}")

    secrets = json.loads(secrets_path.read_text())

    populated = {}
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid manifest line (expected 'KEY: $.path'): '{line}'")

        env_var, _, raw_path = line.partition(":")
        env_var = env_var.strip()
        raw_path = raw_path.strip()

        if not overwrite and env_var in os.environ:
            continue

        value = _resolve_path(secrets, raw_path)
        os.environ[env_var] = value
        populated[env_var] = value

    return populated
