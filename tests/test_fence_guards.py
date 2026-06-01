"""V-RB1 guard: T17 fenced services do not self-document as stub / partial port.

Per V-RB1, the listed service modules must either be ported onto
OutlineNode + outline_subtree or explicitly fenced. "Fenced" rules out
docstrings / log messages that label the surface as a stub or
partial port — fenced surfaces are deliberately disabled, not
in-progress.

Also confirms:
  - the consuming routes are unmounted (V-RB1 "route-disabled" clause),
  - the recommender/analyzer/categorizer surfaces deleted in T53 are gone
    (importing them raises ModuleNotFoundError).
"""

from __future__ import annotations

import importlib
import inspect
import re

import pytest

import app.main as main_mod
import app.scheduler as scheduler_mod
import app.api.v1.tutor as tutor_api_mod

import app.services.tutor.outline as tutor_outline_mod
import app.services.anki.queries as anki_queries_mod
import app.services.anki.state as anki_state_mod
import app.services.anki.retention as anki_retention_mod
import app.api.v1.anki as anki_api_mod


# Match the literal words "stub" or "partial port" as standalone tokens
# (not "stubs" inside "stubs are fenced" — the V-RB1 check is about
# self-labelling the module's *purpose*, not banning the word).
_STUB_PATTERNS = [
    re.compile(r"\bT14 stub\b", re.IGNORECASE),
    re.compile(r"\bT13 stub\b", re.IGNORECASE),
    re.compile(r"\bT14 partial port\b", re.IGNORECASE),
    re.compile(r"\bpartial port\b", re.IGNORECASE),
    re.compile(r'"""Stub\b'),
    re.compile(r"^Stub —", re.MULTILINE),
    re.compile(r"TODO\(T(?:4|13|14)[^)]*\)"),
    # `<func_name> stub: ...` log/msg patterns — bans the legacy log shape
    # without flagging the allowed "FENCED, not a stub:" disclaimer.
    # Identifier ≥ 3 chars so "a stub:" / "an stub:" do not match.
    re.compile(r"\b[a-z_]{3,} stub: "),
]


# NOTE: analytics_mod left this list in T44 — its mastery rollup was ported
# onto OutlineNode + outline_subtree and re-exposed under
# /api/v1/outline/.../mastery (no longer fenced). The recommender + analyzer
# surfaces left in T53 — they were DELETED, not fenced (see
# test_deleted_legacy_surfaces_absent below). anki_queries_mod left in T53 too:
# its FENCED subtree helpers were deleted with topic_subtree, leaving only the
# live outline-free helpers — so it no longer self-documents as fenced (it
# stays in _VRB2_MODULES for the legacy-SQL absence check).
_FENCED_MODULES = [
    tutor_outline_mod,
    anki_state_mod,
    anki_retention_mod,
]


# V-RB2 SQL-pattern absence check — these substrings indicate live raw-SQL
# joins against the dropped `topics` / `content_categories` tables, or
# legacy column references on `anki_note_tags`. They must not appear in
# active code inside the V-RB2-scoped anki services.
_FORBIDDEN_LEGACY_SQL = [
    "FROM topics",
    "JOIN topics",
    "FROM content_categories",
    "JOIN content_categories",
    "topics.parent_topic_id",
    "t.topic_id",
    "tp.topic_id",
    "t.content_category_id",
    "tp.content_category_id",
]


_VRB2_MODULES = [anki_queries_mod, anki_state_mod, anki_retention_mod]


@pytest.mark.parametrize("module", _FENCED_MODULES, ids=lambda m: m.__name__)
def test_module_does_not_self_document_as_stub(module):
    """V-RB1: fenced modules ⊥ self-document as stub / partial port."""
    src = inspect.getsource(module)
    hits = [pat.pattern for pat in _STUB_PATTERNS if pat.search(src)]
    assert not hits, f"{module.__name__} still contains stub/partial-port self-doc patterns: {hits}"


@pytest.mark.parametrize("module", _FENCED_MODULES, ids=lambda m: m.__name__)
def test_module_declares_fence(module):
    """V-RB1: fenced modules carry an explicit FENCED marker + a P0.5 task
    reference (T17 fenced rescope-out services; T18 extended the fence to
    the anki queries/state/retention surfaces)."""
    src = inspect.getsource(module)
    assert "FENCED" in src, f"{module.__name__} missing FENCED marker"
    assert ("T17" in src) or ("T18" in src), f"{module.__name__} missing T17/T18 reference"


def test_api_routers_for_fenced_surfaces_unmounted():
    """V-RB1 route-disabled clause — analyzer/recommendations routers are not
    mounted under `/api/v1/*`. (analytics was ported in T44 and re-exposed
    under `/api/v1/outline/.../mastery`; the old `/api/v1/analytics` route is
    deleted, so it stays absent here too.)"""
    paths = {route.path for route in main_mod.app.routes}
    # Sub-routers contribute to `app.routes` flattened; check known paths.
    forbidden_prefixes = (
        "/api/v1/analytics",
        "/api/v1/analyzer",
        "/api/v1/recommendations",
    )
    bad = [p for p in paths if any(p.startswith(pre) for pre in forbidden_prefixes)]
    assert not bad, f"FENCED API routes still mounted: {bad}"


def test_tutor_outline_node_routes_mounted():
    """T22 unfenced the tutor outline surface onto OutlineNode (V-O1/V-O3).

    The legacy AAMC-shaped routes (`/outline/topics/search`, `/outline`) are
    replaced by domain-blind node-keyed routes; they must be live on the
    public API. The legacy decorators must no longer appear in the source
    (commented or otherwise) so that we don't drift back into the
    AAMC-only shape.
    """
    paths = {route.path for route in main_mod.app.routes}
    expected = {
        "/api/v1/tutor/outline/nodes/search",
        "/api/v1/tutor/outline",
        "/api/v1/tutor/outline/nodes/{node_id}/subtree",
    }
    missing = expected - paths
    assert not missing, f"T22 tutor outline node routes not mounted: {sorted(missing)}"

    src = inspect.getsource(tutor_api_mod)
    assert "/outline/topics/search" not in src, (
        "legacy `/outline/topics/search` decorator string should be removed "
        "(T22 replaced with `/outline/nodes/search`)"
    )


@pytest.mark.parametrize("module", _VRB2_MODULES, ids=lambda m: m.__name__)
def test_vrb2_no_legacy_sql_patterns(module):
    """V-RB2: anki queries/state/retention contain no live raw-SQL joins
    against `topics` / `content_categories` and no legacy `t.topic_id`
    / `t.content_category_id` column references."""
    src = inspect.getsource(module)
    hits = [pat for pat in _FORBIDDEN_LEGACY_SQL if pat in src]
    assert not hits, f"{module.__name__} still contains forbidden legacy-SQL patterns: {hits}"


def test_vrb2_anki_routes_disabled():
    """V-RB2 route-disabled clause — `/api/v1/anki/cards` (topic_id) and
    `/api/v1/anki/performance` are unmounted."""
    paths = {route.path for route in main_mod.app.routes}
    forbidden = {"/api/v1/anki/cards", "/api/v1/anki/performance"}
    bad = forbidden & paths
    assert not bad, f"FENCED anki routes still mounted: {sorted(bad)}"


def test_vrb2_anki_api_module_drops_fenced_imports():
    """V-RB2: `app.api.v1.anki` no longer imports the FENCED helpers in
    active code (all such imports are commented out)."""
    src = inspect.getsource(anki_api_mod)
    forbidden_imports = [
        "list_cards_for_topic",
        "state_for_cc",
        "state_for_topic",
        "retention_for_cc",
        "retention_for_topic",
    ]
    for name in forbidden_imports:
        for match in re.finditer(re.escape(name), src):
            line_start = src.rfind("\n", 0, match.start()) + 1
            line = src[line_start : match.start()]
            # Allow only commented-out occurrences.
            assert line.lstrip().startswith("#"), (
                f"`{name}` referenced in active code of app.api.v1.anki — "
                f"expected only commented-out occurrences (FENCED)"
            )


def test_scheduler_feature_extraction_absent():
    """T53 — `run_feature_extraction` and `run_categorizer` jobs were deleted,
    not fenced. They must not appear in the scheduler source at all (not even
    commented out)."""
    src = inspect.getsource(scheduler_mod)
    assert "run_feature_extraction" not in src, (
        "run_feature_extraction was deleted in T53 but still appears in scheduler"
    )
    assert "run_categorizer" not in src, (
        "run_categorizer was deleted in T53 but still appears in scheduler"
    )
    assert "run_anki_topic_resolver" not in src, (
        "run_anki_topic_resolver was deleted in T53 but still appears in scheduler"
    )


@pytest.mark.parametrize(
    "module_path",
    [
        "app.services.recommender",
        "app.services.topic_subtree",
        "app.services.analyzer",
        # NOTE: app.services.eval was deleted in T53 (legacy MCAT eval) but
        # revived for RCA-11 as the extraction-quality measurement harness
        # (V-L2). It is a live package again — no longer a fenced surface.
        "app.services.categorizer",
        "app.services.llm.batch",
        "app.api.v1.analyzer",
        "app.api.v1.recommendations",
        "app.services.anki.topic_resolver",
        "app.schemas.recommendations",
    ],
)
def test_deleted_legacy_surfaces_absent(module_path):
    """T53 — legacy MCAT categorizer / analyzer / recommender / topic-resolver
    surfaces were deleted (not fenced). Importing them must fail."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_path)
