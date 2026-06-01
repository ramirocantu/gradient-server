"""T51 — POST /api/v1/pdf/ingest contract tests (I.api, V-KB1, V16).

A real (tiny) PDF is rendered by PyMuPDF; the OpenAI client is patched at the
``app.api.v1.pdf.build_openai_client`` seam (V16) to script the vision
transcription then the structured fact-extraction completion.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outline import Course
from tests._openai_mocks import make_client, make_completion

_TOKEN = {"X-Coach-Token": "change_me_before_use"}


def _facts_completion(*facts: str):
    return make_completion(content=json.dumps({"facts": [{"text": f} for f in facts]}))


def _mini_pdf() -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 750), "Lecture page one.")
    data = doc.tobytes()
    doc.close()
    return data


async def _make_course(session: AsyncSession) -> int:
    c = Course(slug=f"pdf-api-{uuid.uuid4().hex[:8]}", name="PDF API Course")
    session.add(c)
    await session.commit()
    return c.id


async def test_pdf_ingest_happy_path(client: AsyncClient, db_session: AsyncSession):
    course_id = await _make_course(db_session)
    pdf = _mini_pdf()

    # One page → one vision call then one extraction call (scripted in order).
    fake = make_client(
        make_completion(content="Lecture page one."),
        _facts_completion("Glycolysis converts glucose to pyruvate."),
    )
    with patch("app.api.v1.pdf.build_openai_client", return_value=fake):
        resp = await client.post(
            "/api/v1/pdf/ingest",
            headers=_TOKEN,
            data={"course_id": str(course_id)},
            files={"file": ("lecture.pdf", pdf, "application/pdf")},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pages"] == 1
    assert body["new_facts"] == 1
    assert body["reused_pdf"] is False
    assert body["extractor_version"] == "pdf-vision-v2"


async def test_pdf_ingest_unknown_course_404(client: AsyncClient):
    pdf = _mini_pdf()
    with patch("app.api.v1.pdf.build_openai_client") as build:
        resp = await client.post(
            "/api/v1/pdf/ingest",
            headers=_TOKEN,
            data={"course_id": "987654"},
            files={"file": ("lecture.pdf", pdf, "application/pdf")},
        )
    assert resp.status_code == 404
    build.assert_not_called()  # rejected before building a client / ingesting


async def test_pdf_ingest_requires_token(client: AsyncClient):
    resp = await client.post(
        "/api/v1/pdf/ingest",
        data={"course_id": "1"},
        files={"file": ("lecture.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 401
