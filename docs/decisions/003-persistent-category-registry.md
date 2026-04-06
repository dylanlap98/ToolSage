# ADR 003: Persistent Category Registry in the Log File

**Date:** 2026-04-06
**Status:** Implemented

---

## The Problem

`usage_category` is assigned by an LLM classifier on each scoring run. Without any anchor, the same operation gets different labels across runs: a month-over-month growth calculation was classified as `financial_metric_calculation` in one run and `data_transformation` in another. The prompt included "be consistent" — but consistency is not enforceable across separate LLM invocations with no shared state.

This directly breaks the improvement loop. `sage.improve()` groups calls by category to accumulate divergence signal. If the same operation is split across two category labels, neither reaches the minimum entry threshold and the pattern is never analyzed. The signal is permanently fragmented.

## The Fix

Each log file now contains a `category_registry` object alongside the `entries` array:

```json
{
  "category_registry": {
    "data_exploration":  "Calls that inspect the raw data structure before performing analysis",
    "revenue_aggregation": "Calculating total and average revenue figures from the dataset"
  },
  "entries": [ ... ]
}
```

Before classifying a new call, the scorer passes the current registry to the classifier prompt:

> *MUST match an existing registry category if the call fits one — only use a new label if no existing category applies.*

After classification, if the returned category is not in the registry, it is added with the description the classifier returned. The registry grows naturally as new operation types are encountered and stabilizes once the tool's usage patterns are established.

## Why the Registry Lives in the Log File

The registry is tool-specific and run-history-specific — it describes the actual operations observed for this tool, not a global taxonomy. Keeping it in the log file makes the log self-contained: the category context needed to interpret the entries is stored alongside them. There is no separate config file to sync, no global state to manage across tools.

## The Remaining Ambiguity in Divergence Direction

The category registry solves signal fragmentation but does not resolve a subtler problem: the same divergence direction can indicate two different root causes.

High quality despite low adherence (`output_quality − manifest_adherence > threshold`) means either:
- The manifest is **too restrictive** — the agent found a better approach that the manifest forbids or discourages
- The manifest is **underdocumented** — the agent used something that works but the manifest never described, so adherence scores low by default

Both cases produce the same signal. The current improver diagnoses which applies by reasoning over raw inputs and errors, but this is a judgment call made by an LLM with imperfect information. A future improvement would be to surface this ambiguity explicitly in the pattern extraction output and require the proposer to state which case it diagnosed before writing a fix.
