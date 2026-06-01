"""T51 — PDF inbox poller tests (V-KB1, V41).

``poll_inbox`` walks ``<inbox>/<course_slug>/*.pdf``, routes each file to its
course, and vision-ingests it. The page renderer is forged and the OpenAI
clients are mocked (V16) so no real PDF or API is touched. V41: an unknown
slug is skipped (not an error) and a per-file render failure is isolated into
``report.failures`` without aborting the batch.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.outline import Course
from app.services.kb.inbox import poll_inbox
from app.services.kb.pdf_ingest import EXTRACTOR_VERSION, RenderedPage
from tests._openai_mocks import make_client, make_completion


def _facts_client(*facts: str):
    return make_client(make_completion(content=json.dumps({"facts": [{"text": f} for f in facts]})))


def _vision_client():
    return make_client(make_completion(content="transcribed page text"))


def _one_page_renderer(_: Path) -> list[RenderedPage]:
    return [RenderedPage(page=1, image_png=b"png-bytes")]


async def _make_course(session: AsyncSession, slug: str) -> Course:
    c = Course(slug=slug, name=slug)
    session.add(c)
    await session.flush()
    return c


def _write_pdf(inbox: Path, slug: str, name: str) -> None:
    d = inbox / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(f"%PDF-1.4 {slug}/{name}".encode())


async def test_poll_ingests_per_slug(db_session: AsyncSession, tmp_path: Path):
    slug = f"bio-{uuid.uuid4().hex[:6]}"
    await _make_course(db_session, slug)
    _write_pdf(tmp_path, slug, "lecture-1.pdf")

    report = await poll_inbox(
        db_session,
        vision_client=_vision_client(),
        extract_client=_facts_client("Insulin is secreted by pancreatic beta cells."),
        inbox_dir=tmp_path,
        renderer=_one_page_renderer,
    )

    assert report.files_seen == 1
    assert report.files_ingested == 1
    assert report.files_skipped == 0
    assert report.new_facts == 1
    assert not report.partial_failure

    facts = (await db_session.execute(select(AtomicFact))).scalars().all()
    assert len(facts) == 1
    assert facts[0].extractor_version == EXTRACTOR_VERSION
    assert facts[0].node_id is None


async def test_poll_skips_unknown_slug(db_session: AsyncSession, tmp_path: Path):
    # No Course with this slug → file skipped, not an error (V41).
    _write_pdf(tmp_path, "no-such-course", "orphan.pdf")

    report = await poll_inbox(
        db_session,
        vision_client=_vision_client(),
        extract_client=_facts_client("never extracted"),
        inbox_dir=tmp_path,
        renderer=_one_page_renderer,
    )

    assert report.files_seen == 1
    assert report.files_skipped == 1
    assert report.files_ingested == 0
    assert report.new_facts == 0


async def test_poll_isolates_per_file_failure(db_session: AsyncSession, tmp_path: Path):
    slug = f"chem-{uuid.uuid4().hex[:6]}"
    await _make_course(db_session, slug)
    _write_pdf(tmp_path, slug, "good.pdf")
    _write_pdf(tmp_path, slug, "bad.pdf")

    def _renderer(path: Path) -> list[RenderedPage]:
        if path.name == "bad.pdf":
            raise ValueError("corrupt PDF")
        return [RenderedPage(page=1, image_png=b"png")]

    report = await poll_inbox(
        db_session,
        vision_client=_vision_client(),
        extract_client=_facts_client("A durable atomic fact here."),
        inbox_dir=tmp_path,
        renderer=_renderer,
    )

    assert report.files_seen == 2
    assert report.files_ingested == 1
    assert report.partial_failure
    assert len(report.failures) == 1
    assert "bad.pdf" in report.failures[0]


async def test_poll_empty_inbox_is_noop(db_session: AsyncSession, tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    report = await poll_inbox(
        db_session,
        vision_client=_vision_client(),
        extract_client=_facts_client("x"),
        inbox_dir=missing,
        renderer=_one_page_renderer,
    )
    assert report.files_seen == 0
    assert not report.partial_failure
