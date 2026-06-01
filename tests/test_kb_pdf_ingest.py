"""T54 — app/services/kb/pdf_ingest.py contract tests (V-KB1, V-KB3, V-KB4).

Notes-ingress redesign: every page is rendered to an image and transcribed by
an OpenAI vision call (V-KB3); the transcription feeds a structured-output
fact-extraction call (V-KB4). Both clients are mocked at the SDK boundary
(V16) via ``tests/_openai_mocks.py``; the page renderer is injected so these
tests never touch PyMuPDF — except the one render smoke that opts in.

V-KB1: re-ingesting a file with the same SHA-256 returns the existing
``pdf_sources`` row and writes no new ``atomic_facts``; ``UQ(course_id,
content_hash)`` dedupes a fact that recurs across PDFs in one course.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.outline import Course
from app.models.pdf_source import PdfSource
from app.services.kb.pdf_ingest import (
    _EXTRACT_MAX_DOC_CHARS,
    EXTRACTOR_VERSION,
    FactExtraction,
    IngestReport,
    PageTranscription,
    RenderedPage,
    extract_atomic_facts,
    ingest_pdf,
    transcribe_page,
)
from tests._openai_mocks import client_with_error, make_client, make_completion


async def _make_course(session: AsyncSession) -> Course:
    c = Course(slug=f"pdf-{uuid.uuid4().hex[:8]}", name="PDF Course")
    session.add(c)
    await session.flush()
    return c


def _fake_pdf(path: Path, content: bytes) -> None:
    path.write_bytes(content)


def _forge_renderer(pages: list[RenderedPage]):
    def _renderer(_: Path) -> list[RenderedPage]:
        return pages

    return _renderer


def _facts_completion(*facts: str, **kw):
    """A structured-output completion whose JSON body lists ``facts``; every
    fact defaults to ``kind='concept'`` (the kept kind, V-KB7)."""
    body = json.dumps({"facts": [{"text": f, "kind": "concept"} for f in facts]})
    return make_completion(content=body, **kw)


def _facts_completion_kinds(pairs: list[tuple[str, str]], **kw):
    """A completion with explicit ``(text, kind)`` pairs — for mixed
    concept/context discernment-gate tests (V-KB7)."""
    body = json.dumps({"facts": [{"text": t, "kind": k} for t, k in pairs]})
    return make_completion(content=body, **kw)


# --------------------------------------------------------------------------- #
# transcribe_page / extract_atomic_facts — SDK-boundary mocks (no DB)
# --------------------------------------------------------------------------- #


async def test_transcribe_page_reads_text_and_tokens():
    client = make_client(
        make_completion(
            content="Glycolysis converts glucose to pyruvate.",
            prompt_tokens=900,
            completion_tokens=50,
            cached_tokens=10,
        )
    )
    out = await transcribe_page(b"\x89PNG fake", client=client, model="gpt-4.1-mini")
    assert isinstance(out, PageTranscription)
    assert out.text == "Glycolysis converts glucose to pyruvate."
    assert out.prompt_tokens == 900
    assert out.output_tokens == 50
    assert out.cached_tokens == 10
    client.chat.completions.create.assert_awaited_once()


async def test_extract_atomic_facts_parses_structured_output():
    client = make_client(_facts_completion("Fact one is here.", "Fact two is here."))
    out = await extract_atomic_facts("some page text", client=client, model="gpt-4.1-mini")
    assert isinstance(out, FactExtraction)
    assert out.facts == [("Fact one is here.", "concept"), ("Fact two is here.", "concept")]


async def test_extract_atomic_facts_carries_kind():
    """V-KB7: each fact carries its concept/context discernment kind."""
    client = make_client(
        _facts_completion_kinds([("A durable fact.", "concept"), ("An example setup.", "context")])
    )
    out = await extract_atomic_facts("text", client=client, model="gpt-4.1-mini")
    assert out.facts == [("A durable fact.", "concept"), ("An example setup.", "context")]


async def test_extract_defaults_malformed_kind_to_concept():
    """V-KB7: a missing/invalid kind defaults to 'concept' — never silently
    dropped because the discriminator field was malformed."""
    body = json.dumps(
        {"facts": [{"text": "No kind field."}, {"text": "Bad kind.", "kind": "bogus"}]}
    )
    client = make_client(make_completion(content=body))
    out = await extract_atomic_facts("text", client=client, model="gpt-4.1-mini")
    assert out.facts == [("No kind field.", "concept"), ("Bad kind.", "concept")]


async def test_vision_and_extract_forward_service_tier():
    """V-L5: service_tier is forwarded onto the chat calls when set, omitted by
    default."""
    vc = make_client(make_completion(content="page text"))
    await transcribe_page(b"\x89PNG", client=vc, model="gpt-5.4-mini", service_tier="flex")
    assert vc.chat.completions.create.await_args.kwargs["service_tier"] == "flex"

    ec = make_client(_facts_completion("A fact."))
    await extract_atomic_facts("text", client=ec, model="gpt-5.4-nano")  # default None
    assert "service_tier" not in ec.chat.completions.create.await_args.kwargs


async def test_extract_atomic_facts_blank_text_skips_llm_call():
    # V-KB4: nothing to extract → no API call, empty result.
    client = client_with_error(AssertionError("must not call the model on blank text"))
    out = await extract_atomic_facts("   \n  ", client=client, model="gpt-4.1-mini")
    assert out.facts == []
    client.chat.completions.create.assert_not_called()


async def test_extract_atomic_facts_drops_empty_and_nonstring():
    body = json.dumps(
        {"facts": [{"text": "  Kept fact.  ", "kind": "concept"}, {"text": "   "}, {"text": 5}]}
    )
    client = make_client(make_completion(content=body))
    out = await extract_atomic_facts("text", client=client, model="gpt-4.1-mini")
    assert out.facts == [("Kept fact.", "concept")]


# --------------------------------------------------------------------------- #
# ingest_pdf — DB-backed, both clients mocked, renderer injected
# --------------------------------------------------------------------------- #


async def test_first_ingest_renders_transcribes_extracts(db_session: AsyncSession, tmp_path: Path):
    course = await _make_course(db_session)
    course_id = course.id
    pdf = tmp_path / "lecture-01.pdf"
    _fake_pdf(pdf, b"%PDF-1.4 fake")

    renderer = _forge_renderer(
        [RenderedPage(page=1, image_png=b"png-1"), RenderedPage(page=2, image_png=b"png-2")]
    )
    # Two pages → two vision calls (transcription) + ONE document-level
    # extraction call (V-KB5: page transcriptions concat into one chunk).
    vision_client = make_client(
        make_completion(content="page one text", prompt_tokens=1000, completion_tokens=100),
        make_completion(content="page two text", prompt_tokens=1000, completion_tokens=100),
    )
    extract_client = make_client(
        _facts_completion(
            "Glycolysis converts glucose to pyruvate.",
            "It yields net two ATP per glucose.",
            "The TCA cycle regenerates oxaloacetate.",
            prompt_tokens=500,
            completion_tokens=80,
            cached_tokens=10,
        ),
    )

    report = await ingest_pdf(
        db_session,
        course_id=course_id,
        path=pdf,
        vision_client=vision_client,
        extract_client=extract_client,
        renderer=renderer,
    )

    assert isinstance(report, IngestReport)
    assert report.reused_pdf is False
    assert report.pages == 2
    assert report.new_facts == 3
    assert report.dup_facts == 0
    assert report.dropped_context_facts == 0
    assert report.extractor_version == EXTRACTOR_VERSION
    # V-KB5: vision is per-page (2 calls); extraction is document-level (1 call).
    assert vision_client.chat.completions.create.await_count == 2
    assert extract_client.chat.completions.create.await_count == 1
    # V-L1: tokens summed across all 3 calls (vision 2×1000 + extract 1×500).
    assert report.input_tokens == 2500
    assert report.output_tokens == 280
    assert report.cached_tokens == 10

    pdf_row = (
        await db_session.execute(select(PdfSource).where(PdfSource.id == report.pdf_source_id))
    ).scalar_one()
    assert pdf_row.status == "ingested"
    assert pdf_row.ingested_at is not None
    assert pdf_row.filename == "lecture-01.pdf"

    facts = (
        (
            await db_session.execute(
                select(AtomicFact).where(AtomicFact.pdf_source_id == report.pdf_source_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(facts) == 3
    # V-KB5: both pages concat into one chunk → stamped with the chunk lead page.
    assert {f.page for f in facts} == {1}
    # V-KB4: node_id NULL until the categorizer (T50) runs; V-KB3: version stamped.
    assert all(f.node_id is None for f in facts)
    assert all(f.extractor_version == EXTRACTOR_VERSION for f in facts)


async def test_context_kind_dropped_at_persist(db_session: AsyncSession, tmp_path: Path):
    """V-KB7: facts the model flags 'context' are dropped before persist and
    counted in ``dropped_context_facts``; only 'concept' rows land in the DB."""
    course = await _make_course(db_session)
    course_id = course.id
    pdf = tmp_path / "mixed.pdf"
    _fake_pdf(pdf, b"%PDF-1.4 mixed")
    renderer = _forge_renderer([RenderedPage(page=1, image_png=b"png-1")])

    report = await ingest_pdf(
        db_session,
        course_id=course_id,
        path=pdf,
        vision_client=make_client(make_completion(content="page text")),
        extract_client=make_client(
            _facts_completion_kinds(
                [
                    ("Force equals mass times acceleration.", "concept"),
                    ("Kinetic energy is one half m v squared.", "concept"),
                    ("The page provides an example with V0 = 50 m/s.", "context"),
                ]
            )
        ),
        renderer=renderer,
    )

    assert report.new_facts == 2
    assert report.dropped_context_facts == 1
    assert report.dup_facts == 0

    rows = (
        (
            await db_session.execute(
                select(AtomicFact).where(AtomicFact.pdf_source_id == report.pdf_source_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2
    texts = {r.text for r in rows}
    assert "The page provides an example with V0 = 50 m/s." not in texts


async def test_extraction_chunks_large_document(db_session: AsyncSession, tmp_path: Path):
    """V-KB5: when concatenated transcriptions exceed the char ceiling, extraction
    splits on page boundaries — more than one extract call, results unioned."""
    course = await _make_course(db_session)
    course_id = course.id
    pdf = tmp_path / "big.pdf"
    _fake_pdf(pdf, b"%PDF-1.4 big")
    renderer = _forge_renderer(
        [RenderedPage(page=1, image_png=b"png-1"), RenderedPage(page=2, image_png=b"png-2")]
    )
    # Each page transcription alone is under the ceiling, but the two together
    # exceed it → two chunks → two extraction calls.
    big_text = "x" * (_EXTRACT_MAX_DOC_CHARS - 100)
    vision_client = make_client(
        make_completion(content=big_text),
        make_completion(content=big_text),
    )
    extract_client = make_client(
        _facts_completion("Fact from chunk one."),
        _facts_completion("Fact from chunk two."),
    )

    report = await ingest_pdf(
        db_session,
        course_id=course_id,
        path=pdf,
        vision_client=vision_client,
        extract_client=extract_client,
        renderer=renderer,
    )

    assert extract_client.chat.completions.create.await_count == 2
    assert report.new_facts == 2


async def test_re_ingest_same_file_is_noop(db_session: AsyncSession, tmp_path: Path):
    course = await _make_course(db_session)
    course_id = course.id
    pdf = tmp_path / "lecture-dup.pdf"
    _fake_pdf(pdf, b"%PDF-1.4 stable-bytes")
    renderer = _forge_renderer([RenderedPage(page=1, image_png=b"png-1")])

    def _clients():
        return (
            make_client(make_completion(content="matrix hosts the TCA cycle")),
            make_client(_facts_completion("Mitochondrial matrix hosts the TCA cycle.")),
        )

    v1, e1 = _clients()
    r1 = await ingest_pdf(
        db_session,
        course_id=course_id,
        path=pdf,
        vision_client=v1,
        extract_client=e1,
        renderer=renderer,
    )
    # Second pass: a client that errors if touched proves the SHA short-circuit
    # returns before any render / API work.
    boom = client_with_error(AssertionError("re-ingest must not render or call the model"))
    r2 = await ingest_pdf(
        db_session,
        course_id=course_id,
        path=pdf,
        vision_client=boom,
        extract_client=boom,
        renderer=_forge_renderer([RenderedPage(page=99, image_png=b"never")]),
    )

    assert r1.reused_pdf is False and r1.new_facts == 1
    assert r2.reused_pdf is True
    assert r2.new_facts == 0
    assert r2.pdf_source_id == r1.pdf_source_id

    pdf_rows = (
        (await db_session.execute(select(PdfSource).where(PdfSource.course_id == course_id)))
        .scalars()
        .all()
    )
    assert len(pdf_rows) == 1


async def test_content_hash_dedupes_facts_within_course(db_session: AsyncSession, tmp_path: Path):
    """V-KB1/V-KB4: a fact recurring in a second PDF of the same course maps to
    the existing row via UQ(course_id, content_hash)."""

    course = await _make_course(db_session)
    course_id = course.id
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    _fake_pdf(pdf_a, b"%PDF-1.4 a")
    _fake_pdf(pdf_b, b"%PDF-1.4 b")

    shared = "Insulin is secreted by pancreatic beta cells."
    renderer = _forge_renderer([RenderedPage(page=1, image_png=b"png")])

    r1 = await ingest_pdf(
        db_session,
        course_id=course_id,
        path=pdf_a,
        vision_client=make_client(make_completion(content="t")),
        extract_client=make_client(_facts_completion(shared)),
        renderer=renderer,
    )
    r2 = await ingest_pdf(
        db_session,
        course_id=course_id,
        path=pdf_b,
        vision_client=make_client(make_completion(content="t")),
        extract_client=make_client(_facts_completion(shared)),
        renderer=renderer,
    )

    assert r1.new_facts == 1
    assert r2.new_facts == 0
    assert r2.dup_facts == 1

    chash = hashlib.sha256(shared.encode()).hexdigest()
    rows = (
        (
            await db_session.execute(
                select(AtomicFact).where(
                    AtomicFact.course_id == course_id, AtomicFact.content_hash == chash
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1


async def test_render_smoke_real_pymupdf(db_session: AsyncSession, tmp_path: Path):
    """End-to-end pass over the real PyMuPDF ``render_pages`` (default renderer)
    with mocked OpenAI clients — exercises rasterization without a real API."""

    pytest.importorskip("pymupdf")
    reportlab = pytest.importorskip(
        "reportlab", reason="reportlab not installed; render smoke skipped"
    )
    assert reportlab  # silence unused
    from reportlab.pdfgen import canvas  # type: ignore

    course = await _make_course(db_session)
    course_id = course.id
    pdf = tmp_path / "smoke.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 720, "Mitochondria are the powerhouse of the cell.")
    c.save()

    report = await ingest_pdf(
        db_session,
        course_id=course_id,
        path=pdf,
        # default renderer = real PyMuPDF; OpenAI still mocked.
        vision_client=make_client(make_completion(content="rendered page text")),
        extract_client=make_client(
            _facts_completion("Mitochondria are the powerhouse of the cell.")
        ),
    )
    assert report.reused_pdf is False
    assert report.pages == 1
    assert report.new_facts == 1
    pdf_row = (
        await db_session.execute(select(PdfSource).where(PdfSource.id == report.pdf_source_id))
    ).scalar_one()
    assert pdf_row.status == "ingested"
