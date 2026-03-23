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
        "Short snake_case label (2-4 words) describing the type of operation the agent "
        "attempted. Derived from the tool's manifest and the actual inputs — not matched "
        "to a fixed taxonomy. Should reflect the tool's domain: a calculator call might "
        "be 'unit_conversion', a RAG call might be 'document_qa', a code tool might be "
        "'script_execution'. Be consistent across calls with similar intent."
    ))


_OUTPUT_QUALITY_PROMPT = """\
You are evaluating how useful a tool call result was for completing an agent's task.
Do not consider how the tool was called — only evaluate the usefulness of the result.

AGENT TASK:
{task}

TOOL CALL:
  inputs: {inputs}
  output: {output}

Score output_quality (0.0-1.0): how well did this result serve the task?
  1.0 — output directly and fully answers what the task needs
  0.7 — output is relevant and useful but not complete
  0.4 — output is loosely relevant; partial value for the task
  0.1 — output is present but provides little useful information
  0.0 — no output or output is entirely useless for the task

Return a score and a 1-2 sentence rationale focused only on result usefulness.\
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
You are classifying a tool call into a usage sub-category based on the tool's manifest
and the actual inputs used.

TOOL MANIFEST:
{manifest}

TOOL CALL INPUTS:
{inputs}

Return a single snake_case usage_category label (2-4 words) that describes what TYPE
of operation the agent was attempting. Derive the category from the manifest's described
use cases and the nature of the inputs — do not match to a fixed list.

Be consistent: calls with similar intent should receive the same category label.

The following are examples of well-formed labels across different tool types.
Use them only to understand the format and granularity expected — your label
should reflect THIS tool's domain:

  Search/lookup tools:    person_lookup, concept_search, document_retrieval, event_lookup
  Calculation tools:      arithmetic_calculation, unit_conversion, statistical_summary
  Code/interpreter tools: script_execution, code_generation, error_debugging, data_transformation
  RAG/knowledge tools:    knowledge_retrieval, document_qa, semantic_search, fact_verification
  Data analysis tools:    dataset_aggregation, trend_analysis, correlation_check, chart_generation
  API/integration tools:  api_query, record_fetch, webhook_trigger, auth_request
  File tools:             file_read, file_write, format_conversion, content_extraction
  Communication tools:    message_send, notification_trigger, email_draft, report_generation\
"""


class Scorer:
    def __init__(self, llm=None, batch_size: int = 5):
        base = llm or ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
        self._quality_judge = base.with_structured_output(_QualityScore)
        self._adherence_judge = base.with_structured_output(_AdherenceScore)
        self._category_classifier = base.with_structured_output(_CategoryClassification)
        self._batch_size = batch_size

    def _build_prompts(self, entry: dict, manifest_content: str) -> tuple[str, str, str]:
        inputs_str = json.dumps(entry["inputs"], indent=2)
        output_str = f"ERROR: {entry['error']}" if entry["error"] else entry["output"]
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
            ),
        )

    def _apply_scores(
        self,
        entry: dict,
        quality: _QualityScore,
        adherence: _AdherenceScore,
        category: _CategoryClassification,
    ) -> None:
        entry["usage_category"] = category.usage_category
        entry["output_quality"] = round(quality.output_quality, 3)
        entry["manifest_adherence"] = round(adherence.manifest_adherence, 3)
        entry["score_rationale"] = (
            f"Output: {quality.rationale} "
            f"Adherence: {adherence.rationale}"
        )
        entry["scored_at"] = datetime.now(timezone.utc).isoformat()

    def score_log(self, log_path: Path, manifest_content: str) -> int:
        """Score all unscored entries concurrently. Returns count of entries scored."""
        content = log_path.read_text() if log_path.exists() else ""
        entries = json.loads(content) if content.strip() else []

        to_score = [e for e in entries if "output_quality" not in e]
        if not to_score:
            return 0

        for batch_start in range(0, len(to_score), self._batch_size):
            batch = to_score[batch_start : batch_start + self._batch_size]
            max_workers = len(batch) * 3

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all calls for all entries in the batch at once
                quality_futures = {
                    id(entry): executor.submit(
                        self._quality_judge.invoke, quality_prompt
                    )
                    for entry, (quality_prompt, _, _) in [
                        (e, self._build_prompts(e, manifest_content)) for e in batch
                    ]
                }
                adherence_futures = {
                    id(entry): executor.submit(
                        self._adherence_judge.invoke, adherence_prompt
                    )
                    for entry, (_, adherence_prompt, _) in [
                        (e, self._build_prompts(e, manifest_content)) for e in batch
                    ]
                }
                category_futures = {
                    id(entry): executor.submit(
                        self._category_classifier.invoke, category_prompt
                    )
                    for entry, (_, _, category_prompt) in [
                        (e, self._build_prompts(e, manifest_content)) for e in batch
                    ]
                }

                for entry in batch:
                    eid = id(entry)
                    self._apply_scores(
                        entry,
                        quality_futures[eid].result(),
                        adherence_futures[eid].result(),
                        category_futures[eid].result(),
                    )

        log_path.write_text(json.dumps(entries, indent=2))
        return len(to_score)
