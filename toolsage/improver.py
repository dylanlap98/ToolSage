"""
Improver — analyzes scored tool call logs and proposes manifest updates.

sage.improve() flow per tool:
  1. Load all scored entries; skip if too few exist per category
  2. Group by usage_category and compute divergence stats
  3. For categories with consistent divergence signal:
       a. LLM: extract failure patterns from rationale fields
       b. LLM: propose a targeted section-level manifest edit
  4. Print diff and prompt for approval — nothing is written without consent

Divergence direction drives the type of improvement proposed:
  adherence_low / quality_high  → manifest too restrictive for this usage type
  adherence_high / quality_low  → manifest incomplete for this usage type
  both_low                      → manifest guidance confirmed wrong; reinforce it
  both_high                     → healthy; no change needed
"""

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from toolsage.manifest import ToolManifest


# ── Pydantic response models ──────────────────────────────────────────────────

class _FailurePatterns(BaseModel):
    patterns: list[str] = Field(
        description=(
            "2-4 bullet points describing concrete, structural patterns observed "
            "in the raw inputs and errors. Each bullet must cite specific examples "
            "from the data (e.g. actual inputs that failed or succeeded), not just "
            "abstract descriptions. Contrast failing calls against succeeding ones."
        ),
    )
    root_cause: str = Field(
        description=(
            "One sentence identifying WHY the failures happened at a structural level — "
            "a constraint of how the tool works, a gap in manifest guidance, or a repeated "
            "misuse pattern. This must be a diagnosis, not a restatement of the symptom."
        )
    )
    agent_behavior_change: str = Field(
        description=(
            "One sentence describing the specific change in agent STRATEGY that would "
            "prevent these failures. Focus on what the agent should DO differently, "
            "not just how to reformat an input."
        )
    )
    target_section: str = Field(
        description=(
            "Which manifest section should receive the fix. Choose based on what the "
            "finding describes:\n"
            "  'Known Failure Modes' — a structural constraint or reliable breakage pattern\n"
            "  'Usage Patterns'      — a strategy for getting better results\n"
            "  'Output Interpretation' — how to read or act on what the tool returns\n"
            "  'Intent'              — a fundamental misunderstanding of what the tool does"
        )
    )


class _ManifestEdit(BaseModel):
    section_name: str = Field(
        description=(
            "Exact name of the manifest section to edit, matching the target_section "
            "identified in the pattern analysis."
        )
    )
    proposed_content: str = Field(
        description=(
            "Complete replacement content for the section body (everything after the "
            "## heading line). Preserve all existing valid guidance. Only add or adjust "
            "what the observed patterns directly justify.\n\n"
            "If your addition includes example inputs, those examples must be grounded "
            "in what the data shows actually works — not invented examples that could "
            "fail for the same structural reason as the observed failures.\n\n"
            "If the fix requires a strategy change, state it directly — do not substitute "
            "an input reformulation that dodges the root cause.\n\n"
            "Match the formatting style of the EXISTING bullets in the section exactly — "
            "same indentation, same use (or absence) of bold, same punctuation conventions. "
            "Do not introduce formatting elements (e.g. bold wrappers, sub-bullets) that "
            "do not already appear in the section."
        )
    )
    rationale: str = Field(
        description=(
            "1-2 sentences: what was added/changed, and why this addresses the root cause "
            "rather than just the surface symptom."
        )
    )


# ── Internal stats container ──────────────────────────────────────────────────

@dataclass
class _CategoryStats:
    usage_category: str
    entry_count: int
    mean_quality: float
    mean_adherence: float
    mean_divergence: float
    divergence_direction: Literal[
        "adherence_low_quality_high",
        "adherence_high_quality_low",
        "both_low",
        "both_high",
        "mixed",
    ]


# ── Prompts ───────────────────────────────────────────────────────────────────

_DIVERGENCE_EXPLANATIONS = {
    "adherence_low_quality_high": (
        "Agents are NOT following the manifest guidance but still getting good results. "
        "This suggests the manifest may be too restrictive or prescriptive for this usage type."
    ),
    "adherence_high_quality_low": (
        "Agents ARE following the manifest guidance but still getting poor results. "
        "This suggests the manifest is missing key guidance for this usage type."
    ),
    "both_low": (
        "Agents are NOT following the manifest AND getting poor results. "
        "The current manifest guidance appears correct — it needs to be reinforced or clarified."
    ),
    "both_high": (
        "Agents are following the manifest and getting good results. No change needed."
    ),
    "mixed": (
        "No consistent divergence signal across these calls."
    ),
}

_PATTERN_EXTRACTION_PROMPT = """\
You are a tool-call analyst. Your job is to diagnose why a category of tool calls
is underperforming and identify the specific manifest change that would fix it.

TOOL: {tool_name}
USAGE CATEGORY: {category}
ENTRIES ANALYZED: {count}

PERFORMANCE SUMMARY:
  Mean output_quality:     {mean_quality:.2f}
  Mean manifest_adherence: {mean_adherence:.2f}
  Mean divergence:         {mean_divergence:+.2f}  (quality minus adherence)

DIVERGENCE SIGNAL: {divergence_explanation}

CALLS (inputs → outcome → scores → judge rationale):
{rationale_text}

Your task — work through this in order:

1. CONTRAST failing vs. succeeding calls by looking at the raw inputs and errors directly.
   What is structurally different between inputs that worked and inputs that failed?
   Cite specific examples from the data above.

2. DIAGNOSE the root cause at the tool level, not the input level.
   Is this a structural constraint of how the tool works?
   A gap in what the manifest tells the agent?
   Or a repeated misuse pattern the manifest should prevent?

3. IDENTIFY the agent behavior change that addresses the root cause.
   A behavior change is a change in what the agent does — how it sequences calls,
   what it checks before or after, or what it decides not to do.
   Reformatting an input is not a behavior change if the structural root cause remains.

4. CHOOSE the correct manifest section for the fix:
   Known Failure Modes   → reliable breakage patterns; structural constraints of the tool
   Usage Patterns        → strategies that improve results when the tool is working
   Output Interpretation → how to read or act on what the tool returns
   Intent                → fundamental misunderstanding of what the tool does\
"""

_EDIT_PROPOSAL_PROMPT = """\
You are writing a targeted improvement to a tool manifest based on a diagnosed failure pattern.

CURRENT MANIFEST:
{manifest_content}

CATEGORY: {category} ({count} calls, divergence: {mean_divergence:+.2f})
DIVERGENCE SIGNAL: {divergence_explanation}

DIAGNOSIS:
  Observed patterns:    {patterns_text}
  Root cause:           {root_cause}
  Agent behavior change:{agent_behavior_change}
  Target section:       {target_section}

Write the updated content for the '{target_section}' section.

Rules:
- Address the ROOT CAUSE, not just the surface symptom. If the fix is a strategy change,
  state it directly and clearly — do not substitute an input reformulation that could
  fail for the same structural reason.
- Preserve every existing bullet that is still valid. Only add or adjust what the
  diagnosis justifies. A single well-placed bullet beats rewriting the section.
- If your addition includes example inputs, they must be grounded in what the data shows
  actually works. Do not invent examples that could fail for the same structural reason
  as the observed failures.
- Be concrete and actionable. Vague guidance ("use more specific queries") is worse than
  no guidance — agents will follow it literally and still fail.
- Match the formatting style of the existing bullets exactly — same use (or non-use) of bold,
  same punctuation, same indentation. Look at the BEFORE content and follow it literally.
  Do not introduce formatting elements that do not already appear in the section.\
"""


# ── Improver ──────────────────────────────────────────────────────────────────

class Improver:
    MIN_ENTRIES_PER_CATEGORY: int = 5
    DIVERGENCE_THRESHOLD: float = 0.20

    def __init__(self, llm=None):
        base = llm or ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
        self._pattern_extractor = base.with_structured_output(_FailurePatterns)
        self._edit_proposer = base.with_structured_output(_ManifestEdit)

    # ── Public ────────────────────────────────────────────────────────────────

    def improve_log(
        self,
        log_path: Path,
        manifest: ToolManifest,
        auto_approve: bool = False,
    ) -> int:
        """Analyze a scored log and propose manifest improvements.

        Returns the number of manifest sections updated."""
        content = log_path.read_text() if log_path.exists() else ""
        entries = json.loads(content) if content.strip() else []

        scored = [e for e in entries if "output_quality" in e and "usage_category" in e]
        if not scored:
            print(f"  No scored entries found in {log_path.name} — run sage.score() first.")
            return 0

        # Show category counts vs threshold before filtering
        from collections import Counter
        raw_counts = Counter(e["usage_category"] for e in scored)
        below = {cat: n for cat, n in raw_counts.items() if n < self.MIN_ENTRIES_PER_CATEGORY}
        if below:
            print(f"  Categories below {self.MIN_ENTRIES_PER_CATEGORY}-entry threshold (need more calls):")
            for cat, n in sorted(below.items(), key=lambda x: -x[1]):
                print(f"    {cat}: {n}/{self.MIN_ENTRIES_PER_CATEGORY}")

        category_stats = self._compute_category_stats(scored)
        actionable = [
            s for s in category_stats
            if s.divergence_direction not in ("both_high", "mixed")
            and abs(s.mean_divergence) >= self.DIVERGENCE_THRESHOLD
        ]

        if not actionable:
            if category_stats:
                print(f"  No actionable divergence found — manifest looks healthy.")
            return 0

        updates = 0
        for stats in actionable:
            cat_entries = [e for e in scored if e.get("usage_category") == stats.usage_category]
            tool_name = log_path.stem.removeprefix("log_")

            print(f"\n  Category: '{stats.usage_category}' "
                  f"({stats.entry_count} calls, divergence: {stats.mean_divergence:+.2f})")
            print(f"  Signal:   {_DIVERGENCE_EXPLANATIONS[stats.divergence_direction]}")

            patterns = self._extract_patterns(tool_name, stats, cat_entries)
            edit = self._propose_edit(manifest.content, stats, patterns)

            self._print_diff(stats, edit, patterns, manifest.content)

            if auto_approve or self._prompt_approval():
                updated = self._apply_edit(manifest.content, edit)
                manifest.path.write_text(updated)
                manifest.content = updated
                print(f"  Written → {manifest.path}")
                updates += 1
            else:
                print("  Skipped.")

        return updates

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _compute_category_stats(self, entries: list[dict]) -> list["_CategoryStats"]:
        groups: dict[str, list[dict]] = {}
        for e in entries:
            groups.setdefault(e["usage_category"], []).append(e)

        stats = []
        for category, group in groups.items():
            if len(group) < self.MIN_ENTRIES_PER_CATEGORY:
                continue
            qualities = [e["output_quality"] for e in group]
            adherences = [e["manifest_adherence"] for e in group]
            mean_q = statistics.mean(qualities)
            mean_a = statistics.mean(adherences)
            mean_div = mean_q - mean_a

            if mean_q >= 0.6 and mean_a >= 0.6:
                direction = "both_high"
            elif abs(mean_div) < self.DIVERGENCE_THRESHOLD:
                direction = "mixed"
            elif mean_div > 0:
                direction = "adherence_low_quality_high"
            elif mean_a > mean_q and mean_q < 0.5:
                direction = "adherence_high_quality_low"
            else:
                direction = "both_low"

            stats.append(_CategoryStats(
                usage_category=category,
                entry_count=len(group),
                mean_quality=round(mean_q, 3),
                mean_adherence=round(mean_a, 3),
                mean_divergence=round(mean_div, 3),
                divergence_direction=direction,
            ))

        return sorted(stats, key=lambda s: abs(s.mean_divergence), reverse=True)

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _extract_patterns(
        self,
        tool_name: str,
        stats: _CategoryStats,
        entries: list[dict],
    ) -> _FailurePatterns:
        call_lines = []
        for e in entries:
            inputs_str = json.dumps(e.get("inputs", {}))
            outcome = f"ERROR: {e['error']}" if e.get("error") else f"OK: {str(e.get('output', ''))[:120]}"
            call_lines.append(
                f"- inputs={inputs_str} | {outcome}\n"
                f"  scores: q={e['output_quality']:.2f}, a={e['manifest_adherence']:.2f} | {e.get('score_rationale', '')}"
            )
        rationale_text = "\n".join(call_lines)
        prompt = _PATTERN_EXTRACTION_PROMPT.format(
            tool_name=tool_name,
            category=stats.usage_category,
            count=stats.entry_count,
            mean_quality=stats.mean_quality,
            mean_adherence=stats.mean_adherence,
            mean_divergence=stats.mean_divergence,
            divergence_explanation=_DIVERGENCE_EXPLANATIONS[stats.divergence_direction],
            rationale_text=rationale_text,
        )
        return self._pattern_extractor.invoke(prompt)

    def _propose_edit(
        self,
        manifest_content: str,
        stats: _CategoryStats,
        patterns: _FailurePatterns,
    ) -> _ManifestEdit:
        patterns_text = "\n".join(f"• {p}" for p in patterns.patterns)
        prompt = _EDIT_PROPOSAL_PROMPT.format(
            manifest_content=manifest_content,
            category=stats.usage_category,
            count=stats.entry_count,
            mean_divergence=stats.mean_divergence,
            divergence_explanation=_DIVERGENCE_EXPLANATIONS[stats.divergence_direction],
            patterns_text=patterns_text,
            root_cause=patterns.root_cause,
            agent_behavior_change=patterns.agent_behavior_change,
            target_section=patterns.target_section,
        )
        return self._edit_proposer.invoke(prompt)

    # ── Manifest surgery ──────────────────────────────────────────────────────

    def _apply_edit(self, manifest_content: str, edit: _ManifestEdit) -> str:
        """Replace the body of one manifest section, preserving everything else."""
        heading = f"## {edit.section_name}"

        # Find the section start
        lines = manifest_content.split("\n")
        section_start = None
        for i, line in enumerate(lines):
            if line.strip() == heading:
                section_start = i
                break

        if section_start is None:
            # Section not found — append it
            new_content = manifest_content.rstrip() + f"\n\n{heading}\n{edit.proposed_content}\n"
            return new_content

        # Find where this section ends (next ## heading or end of file)
        section_end = len(lines)
        for i in range(section_start + 1, len(lines)):
            if lines[i].startswith("## "):
                section_end = i
                break

        new_lines = (
            lines[:section_start + 1]           # heading line
            + [""]                               # blank line after heading
            + edit.proposed_content.splitlines() # new body
            + [""]                               # trailing blank
            + lines[section_end:]                # rest of manifest
        )
        return "\n".join(new_lines)

    # ── Display ───────────────────────────────────────────────────────────────

    def _extract_section_body(self, manifest_content: str, section_name: str) -> str | None:
        """Return the body text of a manifest section, or None if not found."""
        lines = manifest_content.split("\n")
        heading = f"## {section_name}"
        start = None
        for i, line in enumerate(lines):
            if line.strip() == heading:
                start = i + 1
                break
        if start is None:
            return None
        body_lines = []
        for line in lines[start:]:
            if line.startswith("## "):
                break
            body_lines.append(line)
        return "\n".join(body_lines).strip()

    def _print_diff(
        self,
        stats: _CategoryStats,
        edit: _ManifestEdit,
        patterns: _FailurePatterns,
        manifest_content: str,
    ) -> None:
        W = 62  # inner width of the diff box

        print(f"\n  Patterns identified:")
        for p in patterns.patterns:
            print(f"    • {p}")
        print(f"\n  Root cause:     {patterns.root_cause}")
        print(f"  Behavior change:{patterns.agent_behavior_change}")
        print(f"  Target section: {patterns.target_section}")
        print(f"\n  Rationale: {edit.rationale}")

        current_body = self._extract_section_body(manifest_content, edit.section_name)

        print(f"\n  ┌─ BEFORE  ## {edit.section_name} {'─' * max(0, W - len(edit.section_name) - 12)}┐")
        if current_body:
            for line in current_body.splitlines():
                print(f"  │ {line}")
        else:
            print(f"  │ (section not found — will be appended)")
        print(f"  └{'─' * (W + 2)}┘")

        print(f"\n  ┌─ AFTER   ## {edit.section_name} {'─' * max(0, W - len(edit.section_name) - 12)}┐")
        for line in edit.proposed_content.splitlines():
            print(f"  │ {line}")
        print(f"  └{'─' * (W + 2)}┘")

    @staticmethod
    def _prompt_approval() -> bool:
        try:
            answer = input("\n  Apply this change? [y/N]: ").strip().lower()
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
