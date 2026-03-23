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
            "2-4 concise bullet points describing consistent patterns observed "
            "across these calls that explain the divergence signal."
        ),
    )
    improvement_focus: str = Field(
        description=(
            "One sentence describing the single most impactful change to the "
            "manifest that would improve future calls in this category."
        )
    )


class _ManifestEdit(BaseModel):
    section_name: str = Field(
        description=(
            "Exact name of the manifest section to edit. Must be one of the "
            "section headings present in the manifest (e.g. 'Usage Patterns', "
            "'Known Failure Modes', 'Output Interpretation', 'Intent')."
        )
    )
    proposed_content: str = Field(
        description=(
            "Complete replacement content for the section body (everything after "
            "the ## heading line). Preserve existing valid guidance — only add or "
            "adjust what the observed patterns justify. Use the same markdown "
            "formatting as the current section."
        )
    )
    rationale: str = Field(
        description="1-2 sentence explanation of what changed and why."
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
You are analyzing tool call performance data to identify improvement opportunities.

TOOL: {tool_name}
USAGE CATEGORY: {category}
ENTRIES ANALYZED: {count}

PERFORMANCE SUMMARY:
  Mean output_quality:    {mean_quality:.2f}
  Mean manifest_adherence:{mean_adherence:.2f}
  Mean divergence:        {mean_divergence:+.2f}  (quality minus adherence)

SIGNAL: {divergence_explanation}

SCORE RATIONALES FROM THESE CALLS:
{rationale_text}

Identify 2-4 concrete, actionable patterns from these rationales that explain the divergence.
Focus on what the manifest guidance does or does not say that could explain these outcomes.
Then state the single most impactful manifest change for this category.\
"""

_EDIT_PROPOSAL_PROMPT = """\
You are proposing a targeted update to a tool manifest based on observed usage data.

CURRENT MANIFEST:
{manifest_content}

CATEGORY: {category} ({count} calls, divergence: {mean_divergence:+.2f})
SIGNAL: {divergence_explanation}

OBSERVED PATTERNS:
{patterns_text}

IMPROVEMENT FOCUS:
{improvement_focus}

Propose ONE targeted edit to a single manifest section. Rules:
- Edit only what the evidence justifies. Preserve all existing valid guidance.
- Small additions beat rewrites. A single clarifying bullet beats rewriting a section.
- adherence_low_quality_high → consider loosening a constraint or adding an allowed exception
- adherence_high_quality_low → add the missing guidance: a failure mode, output note, or usage pattern
- both_low → reinforce or clarify the current guidance — it seems correct but is being ignored

Return the section name and its complete new content (the section body, not including the ## heading).\
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

            self._print_diff(tool_name, stats, edit, patterns)

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
        rationale_text = "\n".join(
            f"- (q={e['output_quality']:.2f}, a={e['manifest_adherence']:.2f}) {e.get('score_rationale', '')}"
            for e in entries
        )
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
            improvement_focus=patterns.improvement_focus,
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

    def _print_diff(
        self,
        tool_name: str,
        stats: _CategoryStats,
        edit: _ManifestEdit,
        patterns: _FailurePatterns,
    ) -> None:
        print(f"\n  Patterns identified:")
        for p in patterns.patterns:
            print(f"    • {p}")
        print(f"\n  Proposed edit → section: '{edit.section_name}'")
        print(f"  Rationale: {edit.rationale}")
        print(f"\n  {'─' * 60}")
        print(f"  PROPOSED CONTENT:")
        for line in edit.proposed_content.splitlines():
            print(f"  {line}")
        print(f"  {'─' * 60}")

    @staticmethod
    def _prompt_approval() -> bool:
        try:
            answer = input("\n  Apply this change? [y/N]: ").strip().lower()
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
