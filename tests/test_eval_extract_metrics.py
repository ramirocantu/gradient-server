"""RCA-11 extraction-quality metric primitives (pure, no LLM/I/O)."""

from __future__ import annotations

from app.services.eval.extract_metrics import (
    before_after_delta,
    fact_yield,
    is_noise,
    keyword_noise_fraction,
    summarize,
)

# Real noise emitted by pdf-vision-v1 (from the RCA-11 issue) — must flag.
_NOISE = [
    "The page provides an example with V0 = 50 m/s.",
    "In the second example, the distance d is requested.",
    "The text states that a student might forget that KE still has energy in it.",
    "A formula example for maximum height uses Vyf = 0.",
]
# Durable, source-independent facts — must NOT flag.
_CLEAN = [
    "Force equals mass times acceleration.",
    "Kinetic energy equals one half m v squared.",
    "Acceleration due to gravity near Earth's surface is 9.8 m/s^2.",
]


def test_is_noise_flags_lecture_context_phrasing():
    assert all(is_noise(f) for f in _NOISE)


def test_is_noise_passes_durable_facts():
    assert not any(is_noise(f) for f in _CLEAN)


def test_fact_yield_counts():
    assert fact_yield(_CLEAN) == 3
    assert fact_yield([]) == 0


def test_keyword_noise_fraction():
    assert keyword_noise_fraction([]) == 0.0
    assert keyword_noise_fraction(_CLEAN) == 0.0
    assert keyword_noise_fraction(_NOISE) == 1.0
    # 1 noise of 4 → 0.25
    assert keyword_noise_fraction(_CLEAN + _NOISE[:1]) == 0.25


def test_summarize_splits_concept_and_context():
    facts = [(t, "concept") for t in _CLEAN] + [(t, "context") for t in _NOISE]
    res = summarize("model-x", facts)
    assert res.total == 7
    assert res.kept == 3
    assert res.dropped_context == 4
    # noise_fraction is over KEPT facts only — all kept are clean.
    assert res.noise_fraction == 0.0


def test_summarize_noise_over_kept_only():
    # A noisy fact mislabeled 'concept' still counts against the kept noise frac.
    facts = [
        ("Force equals mass times acceleration.", "concept"),
        ("The page provides an example with V0 = 50 m/s.", "concept"),
    ]
    res = summarize("model-x", facts)
    assert res.kept == 2
    assert res.noise_fraction == 0.5


def test_before_after_delta():
    before = summarize("v1", [(t, "concept") for t in _CLEAN + _NOISE])
    after = summarize("v2", [(t, "concept") for t in _CLEAN])
    delta = before_after_delta(before, after)
    assert delta.yield_delta == -4
    assert delta.noise_delta < 0  # noise fraction fell
    assert delta.as_dict()["before"]["kept"] == 7
