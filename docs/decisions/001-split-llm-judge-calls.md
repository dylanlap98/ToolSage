# ADR 001: Split LLM Judge Into Two Independent Calls

**Date:** 2026-03-23
**Status:** Implemented

---

## The Problem

ToolSage's scoring layer evaluates each tool call along two dimensions:

- **output_quality** — how useful was the result for completing the agent's task?
- **manifest_adherence** — did the agent's inputs follow the tool's manifest guidance?

These dimensions need to be independent. Their divergence is the signal the SHAP layer will use to identify where a manifest needs improvement: if an agent ignores the manifest and still gets great results, that's evidence the manifest guidance is too restrictive or wrong. If an agent follows the manifest perfectly and still gets poor results, the manifest is missing something.

The initial implementation asked a single LLM judge to score both dimensions in one call. In practice, this produced scores that were too correlated. When a tool call resulted in an error, the judge would score both `output_quality` and `manifest_adherence` low, anchoring adherence to the outcome. In one illustrative case, a query for `"AlexNet ImageNet 2012"` returned a PageError and received `output_quality: 0.1, manifest_adherence: 0.1`. But this query used three proper nouns, which *partially* follows the manifest's guidance to use concise proper noun queries. The adherence score should have been closer to 0.4, reflecting partial compliance. The agent combined too many terms but wasn't ignoring the manifest entirely.

The single-call design couldn't escape this. The judge had the error in view when scoring adherence, and that context contaminated the evaluation.

## The Fix

Two separate LLM calls per entry, each with only the context it needs:

**Call 1 — output_quality:**
Receives the agent task, the inputs, and the output (or error). Does **not** receive the manifest. The judge has no way to penalize for format violations, it can only evaluate whether the result was useful.

**Call 2 — manifest_adherence:**
Receives the manifest and the inputs only. Does **not** receive the output or error. The judge has no knowledge of whether the call succeeded or failed, it can only evaluate whether the inputs followed the guidance.

This structural isolation guarantees independence. A judge scoring adherence literally cannot see the result, so it cannot be anchored by it.

## The Trade-off

Two LLM calls per entry doubles the scoring cost. For a framework using a small, cheap model (Claude Haiku) for judging, this is acceptable. The data quality improvement is worth it. If cost becomes a concern at scale, batching or caching identical manifest evaluations across runs is a straightforward optimization.

## Why This Matters Downstream

The SHAP analysis layer will look for patterns across many scored calls, specifically cases where `output_quality` and `manifest_adherence` diverge consistently. That divergence is weak signal in any single run but becomes strong signal across hundreds of calls. Correlated scores would make the divergence invisible to SHAP entirely, defeating the purpose of having two dimensions. Separating the calls is a prerequisite for the self-improvement loop to work.
