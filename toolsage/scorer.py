"""
Scorer — LLM-as-judge for tool call quality.

Reads a tool's log file, scores each unscored entry against the tool's manifest
and the agent's task using TWO separate LLM calls per entry — one per dimension.
This guarantees the scores are independent and cannot influence each other.

  output_quality call:     task + inputs + output/error   (no manifest)
  manifest_adherence call: manifest + inputs only          (no output/error)

Each entry gets four new fields:
    output_quality      float 0.0-1.0  how useful was the result for the task?
    manifest_adherence  float 0.0-1.0  did the inputs follow the manifest guidance?
    score_rationale     str            explanation of both scores
    scored_at           str            ISO timestamp

Divergence between the two scores is the signal the SHAP layer uses:
  adherence low,  quality high → manifest guidance may be too restrictive or wrong
  adherence high, quality low  → manifest guidance may be missing something
"""

import json
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


class Scorer:
    def __init__(self, llm=None):
        base = llm or ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
        self._quality_judge = base.with_structured_output(_QualityScore)
        self._adherence_judge = base.with_structured_output(_AdherenceScore)

    def score_log(self, log_path: Path, manifest_content: str) -> int:
        """Score all unscored entries in a log file. Returns count of entries scored."""
        content = log_path.read_text() if log_path.exists() else ""
        entries = json.loads(content) if content.strip() else []

        to_score = [e for e in entries if "output_quality" not in e]
        if not to_score:
            return 0

        for entry in to_score:
            inputs_str = json.dumps(entry["inputs"], indent=2)
            output_str = f"ERROR: {entry['error']}" if entry["error"] else entry["output"]

            quality: _QualityScore = self._quality_judge.invoke(
                _OUTPUT_QUALITY_PROMPT.format(
                    task=entry.get("task") or "(no task context provided)",
                    inputs=inputs_str,
                    output=output_str,
                )
            )

            adherence: _AdherenceScore = self._adherence_judge.invoke(
                _MANIFEST_ADHERENCE_PROMPT.format(
                    manifest=manifest_content,
                    inputs=inputs_str,
                )
            )

            entry["output_quality"] = round(quality.output_quality, 3)
            entry["manifest_adherence"] = round(adherence.manifest_adherence, 3)
            entry["score_rationale"] = (
                f"Output: {quality.rationale} "
                f"Adherence: {adherence.rationale}"
            )
            entry["scored_at"] = datetime.now(timezone.utc).isoformat()

        log_path.write_text(json.dumps(entries, indent=2))
        return len(to_score)
