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
After each tool call, ToolSage scores the interaction - did the agent use the tool correctly? Did the output serve the agent's actual goal? Scoring is lightweight and pluggable (LLM-as-judge by default).
3. Self-Improvement
Scores accumulate. SHAP analysis identifies which factors - prompt wording, context, task description - drove good or bad outcomes. The manifest updates over time. The next agent that calls this tool benefits from every prior call.

---

### How It Works
```
Agent calls tool
      ↓
Manifest injected into agent context (just-in-time)
      ↓
Tool executes
      ↓
Outcome scored (LLM-as-judge or custom scorer)
      ↓
SHAP analysis — which input factors drove the outcome?
      ↓
Call embedded + stored with score
      ↓
Manifest updated if consistent improvement signal found
      ↓
Next agent call retrieves semantically similar past calls
and receives an improved manifest
```

---

### Roadmap
- [x] Core manifest loading and injection
- [x] LLM-as-judge scorer (concurrent, 3 independent calls per entry)
- [x] Usage sub-category classification (enables per-category trend analysis)
- [x] Manifest auto-update loop (`sage.improve()` — divergence-driven, human-in-the-loop)
- [ ] SHAP feature attribution
- [ ] Embedding store for past calls
- [ ] LangGraph integration
- [ ] AWS AgentCore Strands integration
- [ ] MCP tool support
- [ ] OpenAI function calling support

---

### Docs

Design decisions and implementation notes live in [`docs/decisions/`](docs/decisions/).

| # | Decision |
|---|---|
| [001](docs/decisions/001-split-llm-judge-calls.md) | Split LLM judge into two independent calls to isolate output quality from manifest adherence |

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
