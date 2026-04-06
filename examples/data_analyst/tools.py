"""
ToolSage setup and python_repl tool definition.

The tool executes arbitrary Python code in a controlled namespace that
includes SALES_DATA and the stdlib statistics module. Only what is
explicitly provided in the namespace is available — no numpy, pandas, etc.
"""

import io
import statistics as _statistics
import sys
from contextlib import redirect_stdout
from pathlib import Path

# Repo root → toolsage and utilities importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.tools import tool

from toolsage import ToolSage
from utilities.env import populate_env
from data import SALES_DATA

populate_env()

MANIFEST_PATH = str(
    Path(__file__).parent.parent.parent / "manifests" / "python_repl.manifest.md"
)

sage = ToolSage(log_dir=str(Path(__file__).parent / "logs"))


def _namespace() -> dict:
    """Execution namespace available to every python_repl call."""
    return {
        "SALES_DATA": SALES_DATA,
        "statistics": _statistics,
        "__builtins__": __builtins__,
    }


@tool
@sage.tool(MANIFEST_PATH)
def python_repl(code: str) -> str:
    """Execute Python code and return printed output."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(code, _namespace())
        output = buf.getvalue()
        return output if output.strip() else "(no output printed)"
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {e}")
