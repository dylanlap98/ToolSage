# Tool: python_repl

## Intent
Execute a Python code snippet and return whatever was printed to stdout.
Use this tool to compute answers that require calculation or data manipulation.

## Available Scope
```
SALES_DATA["months"]     — 12 month name strings (Jan-Dec)
SALES_DATA["revenue"]    — 12 monthly revenue integers (USD)
SALES_DATA["units_sold"] — 12 monthly unit count integers
statistics               — Python stdlib statistics module
```

## Usage Patterns
- Use `print()` to produce output
- One computation per call

## Known Failure Modes
- Only stdlib is available — third-party packages will raise ImportError
