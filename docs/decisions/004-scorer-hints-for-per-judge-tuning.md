# ADR 004: scorer_hints — Per-Judge Prompt Tuning at the Decorator Level

**Date:** 2026-06-28
**Status:** Implemented

---

## The Problem

The scorer's LLM judges use generic prompts that work across tool types. For most tools, this is fine. But some tools have domain-specific failure modes that the generic prompts misinterpret in a consistent, predictable way.

One concrete case: a `python_repl` tool that executes code in a fresh namespace per call. When the agent references a variable it didn't fetch in the same call, the result is a `NameError`. The quality judge sees an error and scores `output_quality: 0.0` — total failure. But the code logic was correct; the call just had incomplete setup. A score of 0.3-0.5 would be accurate. The generic prompt has no way to know this.

The manifest is the obvious place to put this context — but the manifest gets injected into the agent's docstring. Judge-facing guidance placed there becomes noise in the agent's context window. There is no separate channel for caller-supplied scorer context.

## The Fix

An optional `scorer_hints` parameter on `@sage.tool()`, typed as `dict[str, str]` with keys `"quality"`, `"adherence"`, and `"category"`. Each hint is appended to its judge's prompt only — never injected into the agent's tool description.

```python
@sage.tool(
    "manifests/python_repl.manifest.md",
    scorer_hints={
        "quality": (
            "A NameError from referencing a variable not set up in the same call "
            "is incomplete setup, not a logic error. Score 0.3-0.5, not 0.0."
        ),
    },
)
def python_repl(code: str) -> str:
    ...
```

The hint travels with the tool registration and is passed into `score_log()` at scoring time. The agent never sees it.

## When to Use This

`scorer_hints` is for correcting systematic mis-scoring caused by tool-specific behavior the generic prompts cannot anticipate — stateless execution environments, structured error formats, domain-specific output conventions.

It is not a substitute for a well-written manifest. The manifest is still the primary documentation surface and the thing `improve()` edits. `scorer_hints` fills the gap between the agent-facing manifest and the scorer's need for tool-specific judging context.

## The Adherence Hint Caution

The `"adherence"` key requires care. Raising adherence scores artificially compresses the `output_quality − manifest_adherence` divergence that `sage.improve()` uses to detect manifest gaps. If a tool consistently scores low adherence because the manifest genuinely omits something, an adherence hint can mask that signal and prevent `improve()` from ever proposing the correct fix.

Use `"quality"` hints freely. Use `"adherence"` hints only when the judge is structurally wrong — not when the manifest is incomplete.
