# ToolSage

---

### The Problem
Tool calling in LLM agents is fragile. Agents recieve a tool schema, name, parameters, description and that's it. No context on when to use it well, no memory of how it's been used before, no signal on what made past calls succeed or fail.

Every agent starts from zero. The tool never gets smarter.

---

### What ToolSage Does
ToolSage wraps any tool call with three layers:
1. Augmentation
Each tool ships with a .manifest.md - a human and machine readable file describing intent, good usage patterns, known failure modes, and output interpretation guidance. Agents receive this alongside the schema.
2. Scoring
After each tool call, ToolSage scores the interaction across two independent dimensions: did the output serve the specific operation the agent attempted (`output_quality`)? Did the inputs follow the manifest guidance (`manifest_adherence`)? A per-tool category registry ensures calls are grouped consistently across runs for trend analysis.
3. Self-Improvement
Divergence between scores is the signal: high quality despite low adherence means the manifest is too restrictive or underdocumented (the agent found a better path than the manifest describes); high adherence despite low quality means the manifest is incomplete or actively misleading (the agent followed guidance that didn't produce results). `sage.improve()` groups calls by usage category, diagnoses which case applies, and proposes targeted section-level manifest edits — human-approved before any write.

---

### How the Signal Accumulates

Each scored call produces a divergence value: `output_quality − manifest_adherence`. A single diverging call is just a data point — no action is taken.

Calls are grouped by `usage_category` (e.g. `concept_search`, `correlation_analysis`). Within each category, ToolSage tracks whether divergence is consistent across calls. Two thresholds gate any proposed change:

- **Minimum evidence** — a category must have at least 5 scored calls before analysis runs
- **Minimum divergence** — the mean divergence across those calls must exceed ±0.20

Only when both conditions are met does `sage.improve()` treat the pattern as a signal worth acting on. At that point it diagnoses the root cause, proposes a targeted manifest edit, and asks for approval. A single outlier call never triggers a change — the system requires a pattern.

---

### How It Works
```
Agent calls tool
      ↓
Manifest injected into agent context (just-in-time)
      ↓
Tool executes → call logged (inputs, output, error, duration)
      ↓
sage.score(): LLM-as-judge scores each call independently
  output_quality    — did the output serve the specific operation?
  manifest_adherence — did the inputs follow the manifest?
  usage_category    — what type of operation was this?
      ↓
sage.improve(): divergence analysis by category
  groups calls by usage_category
  computes mean divergence (quality − adherence) per category
  requires ≥5 entries + ≥±0.20 divergence before acting
      ↓
LLM diagnoses root cause from accumulated rationale text
      ↓
LLM proposes targeted manifest section edit
      ↓
Human approves → manifest updated
      ↓
Next agent run uses the improved manifest
```

---

### Roadmap
- [x] Core manifest loading and injection
- [x] LLM-as-judge scorer (concurrent, 3 independent calls per entry)
- [x] Usage sub-category classification with persistent category registry (consistent grouping across runs)
- [x] Manifest auto-update loop (`sage.improve()` — divergence-driven, human-in-the-loop)
- [ ] PyPI package

---

### Docs

Design decisions and implementation notes live in [`docs/decisions/`](docs/decisions/).

| # | Decision |
|---|---|
| [001](docs/decisions/001-split-llm-judge-calls.md) | Split LLM judge into two independent calls to isolate output quality from manifest adherence |
| [002](docs/decisions/002-output-quality-scoped-to-immediate-operation.md) | Score output quality against the immediate tool operation, not the session-level task |
| [003](docs/decisions/003-persistent-category-registry.md) | Persist a category registry in each log file to enforce consistent classification across runs |

---

### Quick Start

```python
from toolsage import ToolSage

sage = ToolSage(log_dir="logs")

@sage.tool("manifests/search_docs.manifest.md")
def search_docs(query: str) -> str:
    # your tool logic
    ...

# 1. Run your agent — manifest is injected JIT, every call is logged
sage.set_task("Find the answer to X")
result = search_docs("how do I reset my password")

# 2. Score logged calls (LLM-as-judge, concurrent)
sage.score()

# 3. Analyze divergence and propose manifest improvements
sage.improve()
```
