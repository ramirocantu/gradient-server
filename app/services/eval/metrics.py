"""Metric primitives for the V-L2 measurement harness.

Pure functions — no I/O, no LLM calls — so the harness can be unit-tested
end-to-end while the actual eval data sits in `tests/fixtures/`. Live runners
(`scripts/run_*_eval.py`) glue these to real LLM calls.

Ported from the phase-3 worktree (tagging-quality gate); the extraction-quality
primitives for RCA-11 live alongside in `extract_metrics.py`.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# V-L2: tagging quality must not regress past the Claude baseline. The gate
# fails if the mean jaccard drops by more than `MEAN_JACCARD_TOLERANCE` OR
# set-equality drops by more than `SET_EQUALITY_TOLERANCE` (both in
# absolute percentage points). The tolerances are small but non-zero — a
# 1-point wobble on a sample-size of ~50 cases is noise.
MEAN_JACCARD_TOLERANCE = 0.03
SET_EQUALITY_TOLERANCE = 0.05


@dataclass(frozen=True)
class EvalCase:
    """One evaluation row.

    - `qid` is opaque (e.g. `"Q1"`) — only used for logging.
    - `gold_tags` is the canonical set of expected node_paths.
    - `predicted_tags` is the set produced by the model under test.
    """

    qid: str
    gold_tags: frozenset[str]
    predicted_tags: frozenset[str]


@dataclass(frozen=True)
class EvalRunResult:
    """Aggregate over an entire eval pass on one model."""

    model: str
    n_cases: int
    mean_jaccard: float
    set_equality_rate: float
    per_case_jaccard: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_cases": self.n_cases,
            "mean_jaccard": round(self.mean_jaccard, 4),
            "set_equality_rate": round(self.set_equality_rate, 4),
            "per_case_jaccard": [round(j, 4) for j in self.per_case_jaccard],
        }


@dataclass(frozen=True)
class EvalReport:
    """Comparison of the model-under-test vs the Claude baseline."""

    baseline: EvalRunResult
    candidate: EvalRunResult
    mean_jaccard_delta: float
    set_equality_delta: float
    blocks_pivot: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.as_dict(),
            "candidate": self.candidate.as_dict(),
            "mean_jaccard_delta": round(self.mean_jaccard_delta, 4),
            "set_equality_delta": round(self.set_equality_delta, 4),
            "blocks_pivot": self.blocks_pivot,
            "reason": self.reason,
        }


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity over two tag sets. Empty-on-both ⇒ 1.0 (perfect
    agreement when there's nothing to disagree about)."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


def record_eval_run(model: str, cases: list[EvalCase]) -> EvalRunResult:
    """Aggregate per-case jaccard + set-equality into a single run result."""
    if not cases:
        return EvalRunResult(
            model=model,
            n_cases=0,
            mean_jaccard=0.0,
            set_equality_rate=0.0,
            per_case_jaccard=(),
        )
    per_case = tuple(jaccard(c.gold_tags, c.predicted_tags) for c in cases)
    set_equality_rate = sum(
        1 for c in cases if c.gold_tags == c.predicted_tags
    ) / len(cases)
    return EvalRunResult(
        model=model,
        n_cases=len(cases),
        mean_jaccard=statistics.fmean(per_case),
        set_equality_rate=set_equality_rate,
        per_case_jaccard=per_case,
    )


def regression_blocks_pivot(
    baseline: EvalRunResult,
    candidate: EvalRunResult,
    *,
    mean_jaccard_tolerance: float = MEAN_JACCARD_TOLERANCE,
    set_equality_tolerance: float = SET_EQUALITY_TOLERANCE,
) -> EvalReport:
    """V-L2: candidate must not drop more than the tolerance on either metric.

    Returns an `EvalReport` with `blocks_pivot=True` and a human-readable
    `reason` when either delta exceeds the tolerance.
    """
    mean_delta = candidate.mean_jaccard - baseline.mean_jaccard
    set_delta = candidate.set_equality_rate - baseline.set_equality_rate

    # Fail closed on incomparable samples — deltas over different case counts
    # don't mean anything, so never let a mismatch pass the gate.
    if baseline.n_cases != candidate.n_cases:
        return EvalReport(
            baseline=baseline,
            candidate=candidate,
            mean_jaccard_delta=mean_delta,
            set_equality_delta=set_delta,
            blocks_pivot=True,
            reason=(
                f"n_cases mismatch: baseline={baseline.n_cases}, "
                f"candidate={candidate.n_cases}"
            ),
        )

    reasons: list[str] = []
    if mean_delta < -mean_jaccard_tolerance:
        reasons.append(
            f"mean_jaccard dropped {mean_delta:+.3f} (tolerance "
            f"-{mean_jaccard_tolerance:.3f})"
        )
    if set_delta < -set_equality_tolerance:
        reasons.append(
            f"set_equality_rate dropped {set_delta:+.3f} (tolerance "
            f"-{set_equality_tolerance:.3f})"
        )

    return EvalReport(
        baseline=baseline,
        candidate=candidate,
        mean_jaccard_delta=mean_delta,
        set_equality_delta=set_delta,
        blocks_pivot=bool(reasons),
        reason="; ".join(reasons) if reasons else "within tolerance",
    )


def load_baseline(path: Path) -> EvalRunResult:
    """Read a baseline JSON file produced by `EvalRunResult.as_dict()`."""
    raw = json.loads(path.read_text())
    return EvalRunResult(
        model=str(raw["model"]),
        n_cases=int(raw["n_cases"]),
        mean_jaccard=float(raw["mean_jaccard"]),
        set_equality_rate=float(raw["set_equality_rate"]),
        per_case_jaccard=tuple(float(x) for x in raw.get("per_case_jaccard", [])),
    )


def write_report(path: Path, report: EvalReport) -> None:
    """Write the JSON report next to the script so it's diffable in PRs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "V-L2 report written: blocks_pivot=%s reason=%s -> %s",
        report.blocks_pivot,
        report.reason,
        path,
    )
