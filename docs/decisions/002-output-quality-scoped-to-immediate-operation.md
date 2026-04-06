# ADR 002: Output Quality Scored Against the Immediate Operation, Not the Session Task

**Date:** 2026-04-06
**Status:** Implemented

---

## The Problem

`output_quality` is meant to measure how useful a tool call result was. The initial prompt asked the judge: *"how well did this result serve the task?"* — where `task` was the full session-level goal set by `sage.set_task()`.

In multi-step agent workflows, this produces systematically wrong scores. A tool call that correctly computed a Pearson correlation coefficient scored 0.2 because the judge evaluated it against a 5-question analysis task and found it only answered one of five questions. The judge's reasoning was technically correct given what it was asked — but the question was wrong.

The issue is architectural: a single tool call in a multi-step workflow is never expected to complete the full task. Penalizing it for not doing so conflates the scope of a tool call with the scope of an agent run.

## The Fix

`task` is relabelled as "context only" in the quality prompt. The evaluation target shifts to the immediate operation described in `inputs`:

> *Look at the INPUTS to determine what specific operation this tool call was trying to accomplish, then evaluate whether the OUTPUT successfully completed THAT operation.*

The scoring scale anchors were also rewritten from "serves the task" to "serves the immediate operation in the inputs." An explicit rule was added:

> *A call that correctly executes one step of a multi-step task should score 1.0 if it accomplished what its inputs asked for — the overall task being incomplete is irrelevant to this score.*

## Why This Matters

This distinction is critical for the improvement loop to work correctly. If multi-step tool calls are systematically underscored, `output_quality` trends low across the board regardless of actual tool performance. The divergence signal (`output_quality − manifest_adherence`) becomes noise, and `sage.improve()` fires on phantom problems.

Scoping quality to the immediate operation means the score reflects what the tool actually did — which is the only thing ToolSage can observe and the only thing the manifest can influence.

## The Constraint This Creates

The quality judge must infer the immediate intent from `inputs` alone, without the agent's internal plan. This works well when inputs are descriptive (e.g. code with comments, structured queries) but is harder when inputs are terse. `sage.set_task()` provides domain context that helps the judge interpret ambiguous inputs — it is "context only" but not useless.
