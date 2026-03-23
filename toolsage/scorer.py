"""
Scorer — LLM-as-judge for tool call quality.

Reads a tool's log file, scores each unscored entry against the tool's manifest
and the agent's task, then writes scores back into the log.

Each entry gets four new fields:
    output_quality      float 0.0-1.0  how useful was the result for the task?
    manifest_adherence  float 0.0-1.0  did the inputs follow the manifest guidance?
    score_rationale     str            explanation covering both dimensions
    scored_at           str            ISO timestamp

Keeping these two dimensions separate is intentional — their divergence is the
signal the SHAP layer uses to identify where the manifest needs improvement:
  adherence high, quality low  → manifest guidance may be missing something
  adherence low,  quality high → manifest guidance may be too restrictive or wrong
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field


class _CallScore(BaseModel):
    output_quality: float = Field(ge=0.0, le=1.0, description=(
        "How useful was the actual result for completing the agent's task? "
        "Score purely on outcome — ignore whether the inputs followed the manifest."
    ))
    manifest_adherence: float = Field(ge=0.0, le=1.0, description=(
        "How closely did the inputs follow the manifest's guidance? "
        "Score purely on input behaviour — ignore how good the output was."
    ))
    rationale: str = Field(description=(
        "1-3 sentences. Address both dimensions separately. "
        "Explicitly flag cases where they diverge — e.g. the agent ignored the manifest "
        "but got great results, or followed it perfectly but got poor results."
    ))


_JUDGE_PROMPT_SUCCESS = """\
You are scoring a single tool call made by an AI agent.

AGENT TASK:
{task}

TOOL MANIFEST (defines correct usage, known failure modes, and output guidance):
{manifest}

TOOL CALL:
  inputs: {inputs}
  output: {output}

Return two independent scores and a rationale:

output_quality (0.0-1.0): How useful was the output for completing the agent's task?
  1.0 — output directly and fully serves the task
  0.7 — output is relevant but not ideal
  0.4 — output is only loosely relevant
  0.1 — output is present but unhelpful
  0.0 — output is useless for the task

manifest_adherence (0.0-1.0): How closely did the inputs follow the manifest's guidance?
  1.0 — inputs match manifest guidance exactly
  0.7 — minor deviation from manifest guidance
  0.4 — notable deviation but not a clear violation
  0.1 — inputs clearly violated manifest guidance
  0.0 — inputs ignored the manifest entirely

Score each dimension independently — a call can have high output_quality and low
manifest_adherence, or vice versa. These divergences are important signal.
Explicitly call out any divergence in your rationale.\
"""

_JUDGE_PROMPT_ERROR = """\
You are scoring a single tool call made by an AI agent. This call resulted in an error.

AGENT TASK:
{task}

TOOL MANIFEST (defines correct usage, known failure modes, and output guidance):
{manifest}

TOOL CALL:
  inputs: {inputs}
  output: null
  error:  {error}

Return two independent scores and a rationale:

output_quality (0.0-1.0): How useful was this call for completing the agent's task?
  0.0 — error produced no useful output; task was not served
  0.1 — error message itself gave the agent useful retry guidance
  (Scores above 0.1 are unlikely for errored calls)

manifest_adherence (0.0-1.0): How closely did the inputs follow the manifest's guidance?
  1.0 — inputs matched manifest guidance; the error was unavoidable
  0.4 — inputs were reasonable but triggered a known failure mode in the manifest
  0.1 — inputs violated manifest guidance and the error was a predictable consequence
  0.0 — inputs clearly ignored the manifest

Score each dimension independently. Explicitly note in the rationale whether the
error was caused by manifest non-adherence or was unavoidable given correct usage.\
"""


class Scorer:
    def __init__(self, llm=None):
        base = llm or ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
        self._judge = base.with_structured_output(_CallScore)

    def score_log(self, log_path: Path, manifest_content: str) -> int:
        """Score all unscored entries in a log file. Returns count of entries scored."""
        content = log_path.read_text() if log_path.exists() else ""
        entries = json.loads(content) if content.strip() else []

        to_score = [e for e in entries if "output_quality" not in e]
        if not to_score:
            return 0

        for entry in to_score:
            if entry["error"]:
                prompt = _JUDGE_PROMPT_ERROR.format(
                    task=entry.get("task") or "(no task context provided)",
                    manifest=manifest_content,
                    inputs=json.dumps(entry["inputs"], indent=2),
                    error=entry["error"],
                )
            else:
                prompt = _JUDGE_PROMPT_SUCCESS.format(
                    task=entry.get("task") or "(no task context provided)",
                    manifest=manifest_content,
                    inputs=json.dumps(entry["inputs"], indent=2),
                    output=entry["output"],
                )
            scored: _CallScore = self._judge.invoke(prompt)
            entry["output_quality"] = round(scored.output_quality, 3)
            entry["manifest_adherence"] = round(scored.manifest_adherence, 3)
            entry["score_rationale"] = scored.rationale
            entry["scored_at"] = datetime.now(timezone.utc).isoformat()

        log_path.write_text(json.dumps(entries, indent=2))
        return len(to_score)
