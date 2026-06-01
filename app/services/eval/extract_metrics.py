"""Extraction-quality metric primitives for the RCA-11 measurement harness.

Pure functions — no I/O, no LLM calls — so they unit-test cleanly. The live
runner (`scripts/run_rca11_extract_eval.py`) glues these to real extraction
calls over a snapshotted-transcription fixture.

The headline question RCA-11 measures: did the doc-level learning-goals rework
cut lecture-context noise without gutting genuine fact yield? We can't automate
the human "is this memorizable?" judgment — but we can automate a *proxy*: the
fraction of kept facts that match lecture-context noise phrasing (the same kind
of crude keyword heuristic that flagged 33/124 in the issue). The runner dumps
the kept facts so a human makes the final call (V-L2 stays human-judged).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Crude lecture-context-noise heuristic. Each pattern matches phrasing typical of
# example/figure/meta prose rather than a durable, source-independent fact —
# e.g. "the page provides an example", "in the second example", "the text states",
# "shown in the diagram". This is a PROXY for noise, not ground truth: a low
# fraction is necessary, not sufficient, for quality. Human judgment is final.
_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # The single strongest signal in the issue's sample: example-bound prose.
    re.compile(r"\bexamples?\b", re.I),
    re.compile(r"\bthe (text|page|slide|diagram|figure|author)\b", re.I),
    re.compile(r"\bthis (problem|question|page|slide|figure|diagram)\b", re.I),
    re.compile(r"\b(shown|depicted|illustrated|pictured|displayed|stated|mentioned) "
               r"(in|on|above|below|here)\b", re.I),
    re.compile(r"\b(is|are) (given|requested|asked|provided)\b", re.I),
    re.compile(r"\b(it is stated|the problem asks|a common .* question)\b", re.I),
)


def is_noise(fact: str) -> bool:
    """True if a fact matches the lecture-context-noise heuristic."""
    return any(p.search(fact) for p in _NOISE_PATTERNS)


def fact_yield(facts: list[str]) -> int:
    """Number of facts produced."""
    return len(facts)


def keyword_noise_fraction(facts: list[str]) -> float:
    """Fraction of facts matching the noise heuristic. Empty input ⇒ 0.0."""
    if not facts:
        return 0.0
    return sum(1 for f in facts if is_noise(f)) / len(facts)


@dataclass(frozen=True)
class ExtractRunResult:
    """Summary of one extraction pass over a document fixture.

    - `total` = every fact the model emitted (concept + context).
    - `kept` = facts classified `concept` (what actually persists).
    - `dropped_context` = facts classified `context` (dropped at persist, V-KB7).
    - `noise_fraction` = `keyword_noise_fraction` over the KEPT facts (the proxy
      that should fall toward ~0 after the rework).
    """

    label: str
    total: int
    kept: int
    dropped_context: int
    noise_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "total": self.total,
            "kept": self.kept,
            "dropped_context": self.dropped_context,
            "noise_fraction": round(self.noise_fraction, 4),
        }


def summarize(label: str, facts: list[tuple[str, str]]) -> ExtractRunResult:
    """Roll up `(text, kind)` pairs into an `ExtractRunResult`.

    `noise_fraction` is computed over the kept (`concept`) facts only — that's
    the surface that pollutes the store and downstream tagging.
    """
    kept = [text for text, kind in facts if kind == "concept"]
    dropped = sum(1 for _, kind in facts if kind != "concept")
    return ExtractRunResult(
        label=label,
        total=len(facts),
        kept=len(kept),
        dropped_context=dropped,
        noise_fraction=keyword_noise_fraction(kept),
    )


@dataclass(frozen=True)
class ExtractDelta:
    """Before/after comparison of two extraction passes."""

    before: ExtractRunResult
    after: ExtractRunResult

    @property
    def yield_delta(self) -> int:
        return self.after.total - self.before.total

    @property
    def noise_delta(self) -> float:
        return self.after.noise_fraction - self.before.noise_fraction

    def as_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.as_dict(),
            "after": self.after.as_dict(),
            "yield_delta": self.yield_delta,
            "noise_delta": round(self.noise_delta, 4),
        }


def before_after_delta(
    before: ExtractRunResult, after: ExtractRunResult
) -> ExtractDelta:
    """Pair two runs for a diffable before/after report."""
    return ExtractDelta(before=before, after=after)
