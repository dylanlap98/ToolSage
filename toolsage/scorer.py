"""
Scorer — LLM-as-judge for tool call quality.

Uses THREE separate LLM calls per entry to guarantee score independence:

  output_quality call:     task + inputs + output/error   (no manifest)
  manifest_adherence call: manifest + inputs only          (no output/error)
  usage_category call:     manifest + inputs only          (classifies usage type)

All three calls for a batch of entries are submitted concurrently via ThreadPoolExecutor.
Default batch size is 5 entries × 3 calls = 15 concurrent LLM calls per batch.

Each entry gets five new fields:
    usage_category      str            what type of operation was this?
    output_quality      float 0.0-1.0  how useful was the result for the task?
    manifest_adherence  float 0.0-1.0  did the inputs follow the manifest guidance?
    score_rationale     str            explanation of both scores
    scored_at           str            ISO timestamp

usage_category enables sub-category trend analysis in sage.improve() — the same
tool can behave differently across usage types, so manifest improvements must be
targeted at specific categories rather than applied globally.

Divergence between scores is the signal the improvement layer uses:
  adherence low,  quality high → manifest too restrictive for this usage type
  adherence high, quality low  → manifest incomplete for this usage type
  adherence low,  quality low  → manifest guidance confirmed correct
"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field


class _QualityScore(BaseModel):
    output_quality: float = Field(ge=0.0, le=1.0)
    rationale: str


class _AdherenceScore(BaseModel):
    manifest_adherence: float = Field(ge=0.0, le=1.0)
    rationale: str


class _CategoryClassification(BaseModel):
    usage_category: str = Field(description=(
        "Short snake_case label (2-4 words) for the type of operation. "
        "MUST match an existing registry category if the call fits one — "
        "only use a new label if no existing category applies."
    ))
    category_description: str = Field(description=(
        "One sentence describing what this category represents as a general pattern "
        "across calls of this type. Written generically, not specific to this one call. "
        "If matching an existing category, reproduce its existing description unchanged."
    ))


_OUTPUT_QUALITY_PROMPT = """\
You are evaluating how useful a tool call result was for the specific operation the agent attempted.

AGENT TASK (context only — the broader goal the agent is working toward):
{task}

TOOL CALL:
  inputs: {inputs}
  output: {output}

The agent task above is context. Your evaluation target is narrower:
look at the INPUTS to determine what specific operation this tool call was trying to accomplish,
then evaluate whether the OUTPUT successfully completed THAT operation.

Tools are often called many times across a workflow. A call that correctly executes
one step of a multi-step task should score 1.0 if it accomplished what its inputs asked for —
the overall task being incomplete is irrelevant to this score.

Score output_quality (0.0-1.0): how well did the output serve the immediate operation in the inputs?
  1.0 — output directly and fully satisfies what the inputs asked for
  0.7 — output is useful and mostly satisfies the inputs, with minor gaps
  0.4 — output partially satisfies the inputs; the operation was attempted but incomplete
  0.1 — output has little useful value for what the inputs requested
  0.0 — no output, or output is entirely wrong for the requested operation

Return a score and a 1-2 sentence rationale focused only on whether the output served the inputs.\
"""

_MANIFEST_ADHERENCE_PROMPT = """\
You are evaluating how closely an agent's tool inputs followed the tool's manifest guidance.
Do not consider the result — only evaluate the inputs against the manifest.

TOOL MANIFEST (defines correct usage, known failure modes, and output guidance):
{manifest}

TOOL CALL INPUTS:
{inputs}

Score manifest_adherence (0.0-1.0): how closely did the inputs follow the manifest?
  1.0 — inputs match manifest guidance exactly
  0.7 — minor deviation (e.g. slightly outside recommended format but intent is correct)
  0.4 — partial adherence — some elements followed, others did not
  0.1 — inputs clearly violated key manifest guidance
  0.0 — inputs ignored the manifest entirely

Return a score and a 1-2 sentence rationale focused only on input behaviour vs the manifest.\
"""

_CATEGORY_PROMPT = """\
You are classifying a tool call into a usage sub-category.

TOOL MANIFEST:
{manifest}

TOOL CALL INPUTS:
{inputs}

EXISTING CATEGORY REGISTRY:
{registry_text}

Classification rules:
1. If the call matches an existing registry category, USE THAT EXACT LABEL — do not invent
   a synonym or variant. Consistency across runs is more important than precision.
2. Only create a new category if no existing one fits. New labels must follow the same
   snake_case 2-4 word format and specificity level as the existing registry.
3. Return the category_description: if matching an existing category, copy its description
   exactly. If creating a new one, write one generic sentence describing the pattern.\
"""


class Scorer:
    def __init__(self, llm=None, batch_size: int = 5):
        base = llm or ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
        self._quality_judge = base.with_structured_output(_QualityScore)
        self._adherence_judge = base.with_structured_output(_AdherenceScore)
        self._category_classifier = base.with_structured_output(_CategoryClassification)
        self._batch_size = batch_size

    def _build_prompts(
        self, entry: dict, manifest_content: str, registry: dict
    ) -> tuple[str, str, str]:
        inputs_str = json.dumps(entry["inputs"], indent=2)
        output_str = f"ERROR: {entry['error']}" if entry["error"] else entry["output"]
        registry_text = (
            "\n".join(f"- {k}: {v}" for k, v in registry.items())
            if registry else "(none yet — this is the first call being classified)"
        )
        return (
            _OUTPUT_QUALITY_PROMPT.format(
                task=entry.get("task") or "(no task context provided)",
                inputs=inputs_str,
                output=output_str,
            ),
            _MANIFEST_ADHERENCE_PROMPT.format(
                manifest=manifest_content,
                inputs=inputs_str,
            ),
            _CATEGORY_PROMPT.format(
                manifest=manifest_content,
                inputs=inputs_str,
                registry_text=registry_text,
            ),
        )

    def _apply_scores(
        self,
        entry: dict,
        quality: _QualityScore,
        adherence: _AdherenceScore,
        category: _CategoryClassification,
        registry: dict,
    ) -> None:
        entry["usage_category"] = category.usage_category
        entry["output_quality"] = round(quality.output_quality, 3)
        entry["manifest_adherence"] = round(adherence.manifest_adherence, 3)
        entry["score_rationale"] = (
            f"Output: {quality.rationale} "
            f"Adherence: {adherence.rationale}"
        )
        entry["scored_at"] = datetime.now(timezone.utc).isoformat()
        # Add to registry if this is a new category
        if category.usage_category not in registry:
            registry[category.usage_category] = category.category_description

    def score_log(self, log_path: Path, manifest_content: str) -> int:
        """Score all unscored entries concurrently. Returns count of entries scored."""
        from toolsage.logger import CallLogger
        registry, entries = CallLogger._read(log_path)

        to_score = [e for e in entries if "output_quality" not in e]
        if not to_score:
            return 0

        for batch_start in range(0, len(to_score), self._batch_size):
            batch = to_score[batch_start : batch_start + self._batch_size]
            max_workers = len(batch) * 3

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Build prompts with current registry state (sequential — registry grows each batch)
                prompts = [(e, self._build_prompts(e, manifest_content, registry)) for e in batch]

                quality_futures = {
                    id(entry): executor.submit(self._quality_judge.invoke, quality_prompt)
                    for entry, (quality_prompt, _, _) in prompts
                }
                adherence_futures = {
                    id(entry): executor.submit(self._adherence_judge.invoke, adherence_prompt)
                    for entry, (_, adherence_prompt, _) in prompts
                }
                category_futures = {
                    id(entry): executor.submit(self._category_classifier.invoke, category_prompt)
                    for entry, (_, _, category_prompt) in prompts
                }

                for entry in batch:
                    eid = id(entry)
                    self._apply_scores(
                        entry,
                        quality_futures[eid].result(),
                        adherence_futures[eid].result(),
                        category_futures[eid].result(),
                        registry,
                    )

        CallLogger._write(log_path, registry, entries)
        return len(to_score)
