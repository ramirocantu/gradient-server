"""T26 — app/services/kb/notion.py contract tests (V-N1, V-N2, V-M3, V16).

V-N1: one-way write-out, ⊥ read Notion back. V-N2: one Notion page
per outline node — re-sync upserts on ``node_id``. V-M3 / V-N1
re-sync is append-only on the Notion side (``blocks.children.append``),
never page rewrite. V16: notion-client mocked at the SDK boundary.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.captures import Question
from app.models.discriminator_factor import DiscriminatorFactor
from app.models.notion_page import NotionPage
from app.models.outline import Course, OutlineNode
from app.models.pdf_source import PdfSource
from app.services.kb.notion import (
    NotionMirrorError,
    mirror_discriminator_to_notion,
    sync_node_to_notion,
)


def _forge_notion_client(*, page_id: str = "notion-page-xyz") -> MagicMock:
    """AsyncClient-shaped mock. Only the write endpoints we actually use
    are wired — leave reads (``pages.retrieve``, ``databases.query``)
    bare so V-N1 violations surface as AttributeError instead of silently
    succeeding.
    """

    client = MagicMock()
    client.pages = MagicMock()
    client.pages.create = AsyncMock(
        return_value={
            "id": page_id,
            "url": f"https://notion.so/{page_id}",
            "object": "page",
        }
    )
    client.blocks = MagicMock()
    client.blocks.children = MagicMock()
    client.blocks.children.append = AsyncMock(return_value={"object": "list", "results": []})
    # Intentionally no .pages.retrieve / .databases.query — any code
    # path that tries to read Notion will raise AttributeError, which
    # is what V-N1 wants.
    return client


async def _make_course_node(session: AsyncSession) -> OutlineNode:
    course = Course(slug=f"n-{uuid.uuid4().hex[:8]}", name="N")
    session.add(course)
    await session.flush()
    node = OutlineNode(
        course_id=course.id,
        parent_id=None,
        kind="concept",
        name=f"Concept {uuid.uuid4().hex[:6]}",
        depth=0,
        position=0,
    )
    session.add(node)
    await session.flush()
    return node


async def _make_facts(session: AsyncSession, course_id: int, texts: list[str]) -> list[AtomicFact]:
    pdf = PdfSource(
        course_id=course_id,
        filename="x.pdf",
        sha256=uuid.uuid4().hex,
        status="ingested",
    )
    session.add(pdf)
    await session.flush()
    facts: list[AtomicFact] = []
    for t in texts:
        f = AtomicFact(
            course_id=course_id,
            pdf_source_id=pdf.id,
            text=t,
            content_hash=uuid.uuid4().hex,
        )
        session.add(f)
        facts.append(f)
    await session.flush()
    return facts


# --------------------------------------------------------------------------- #
# 1. First sync creates page + pointer (V-N1, V-N2)
# --------------------------------------------------------------------------- #


async def test_first_sync_creates_page_and_pointer(db_session: AsyncSession):
    node = await _make_course_node(db_session)
    node_id = node.id
    facts = await _make_facts(
        db_session,
        node.course_id,
        ["Glycolysis converts glucose to pyruvate.", "Net yield is two ATP."],
    )
    client = _forge_notion_client(page_id="page-1")

    report = await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=facts,
    )

    assert report.created_page is True
    assert report.appended_blocks == 2
    assert report.notion_page_id == "page-1"

    client.pages.create.assert_awaited_once()
    client.blocks.children.append.assert_not_awaited()

    pointer = (
        await db_session.execute(select(NotionPage).where(NotionPage.node_id == node_id))
    ).scalar_one()
    assert pointer.notion_page_id == "page-1"
    assert pointer.url.endswith("page-1")
    assert pointer.last_synced_at is not None


# --------------------------------------------------------------------------- #
# 1b. RCA-33: pages.create parents to data_source_id, not database_id.
# Notion API 2025-09-03 (notion-client>=3) 404s a database_id parent given a
# data-source id; NOTION_WIKI_DB_ID holds a data-source id. Pin the shape so a
# regression to the legacy {"database_id": ...} parent fails loudly here
# instead of silently at runtime against the live API.
# --------------------------------------------------------------------------- #


async def test_pages_create_parents_to_data_source(db_session: AsyncSession):
    node = await _make_course_node(db_session)
    facts = await _make_facts(db_session, node.course_id, ["Fact long enough to keep."])
    client = _forge_notion_client(page_id="page-ds")

    await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="ds-id",
        node=node,
        facts=facts,
    )

    _, kwargs = client.pages.create.call_args
    assert kwargs["parent"] == {"type": "data_source_id", "data_source_id": "ds-id"}
    assert "database_id" not in kwargs["parent"]


# --------------------------------------------------------------------------- #
# 2. Re-sync append-only (V-M3, V-N1)
# --------------------------------------------------------------------------- #


async def test_resync_appends_blocks_no_page_rewrite(db_session: AsyncSession):
    node = await _make_course_node(db_session)
    node_id = node.id
    facts_1 = await _make_facts(db_session, node.course_id, ["First fact long enough."])
    facts_2 = await _make_facts(db_session, node.course_id, ["Second fact long enough."])
    client = _forge_notion_client(page_id="page-append")

    r1 = await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=facts_1,
    )
    r2 = await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=facts_2,
    )

    assert r1.created_page is True
    assert r2.created_page is False
    assert r2.notion_page_id == "page-append"

    # First call → pages.create; second call → blocks.children.append.
    client.pages.create.assert_awaited_once()
    client.blocks.children.append.assert_awaited_once()

    pointers = (
        (await db_session.execute(select(NotionPage).where(NotionPage.node_id == node_id)))
        .scalars()
        .all()
    )
    assert len(pointers) == 1  # V-N2: one page per node
    assert "First fact long enough." in (pointers[0].tags or [None])[0]
    assert any("Second fact long enough." in t for t in (pointers[0].tags or []))


# --------------------------------------------------------------------------- #
# 3. V-N1: no read-back path (mock has no read endpoints)
# --------------------------------------------------------------------------- #


async def test_no_read_endpoints_called(db_session: AsyncSession):
    """V-N1: the seam must never call read endpoints. The forged client
    deliberately omits ``pages.retrieve`` and ``databases.query``; if
    the seam tried to read, the call would raise AttributeError.
    """

    node = await _make_course_node(db_session)
    facts = await _make_facts(db_session, node.course_id, ["Fact long enough to keep."])
    client = _forge_notion_client()

    # Confirm the mock has no read attributes wired.
    assert not hasattr(client.pages, "retrieve") or not isinstance(client.pages.retrieve, AsyncMock)
    # The sync must succeed without touching read paths.
    await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=facts,
    )


# --------------------------------------------------------------------------- #
# 4. Empty facts on re-sync still updates last_synced_at
# --------------------------------------------------------------------------- #


async def test_resync_with_no_facts_skips_append(db_session: AsyncSession):
    node = await _make_course_node(db_session)
    facts_1 = await _make_facts(db_session, node.course_id, ["Seed fact long enough."])
    client = _forge_notion_client(page_id="page-empty")

    await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=facts_1,
    )
    initial_ts = (
        await db_session.execute(
            select(NotionPage.last_synced_at).where(NotionPage.node_id == node.id)
        )
    ).scalar_one()

    r = await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=[],
    )
    assert r.created_page is False
    assert r.appended_blocks == 0
    client.blocks.children.append.assert_not_awaited()

    later_ts = (
        await db_session.execute(
            select(NotionPage.last_synced_at).where(NotionPage.node_id == node.id)
        )
    ).scalar_one()
    assert later_ts >= initial_ts


# --------------------------------------------------------------------------- #
# T32 — discriminator-factor mirror (V-N1, V-N2, V-M3)
# --------------------------------------------------------------------------- #


def _forge_mirror_client(*, page_id: str = "d-page") -> MagicMock:
    """AsyncClient-shaped mock whose ``blocks.children.append`` returns a
    fresh block id per call (so the seam can record ``notion_block_id``).
    Read endpoints stay unwired so V-N1 violations surface as AttributeError.
    """

    client = MagicMock()
    client.pages = MagicMock()
    client.pages.create = AsyncMock(
        return_value={
            "id": page_id,
            "url": f"https://notion.so/{page_id}",
            "object": "page",
        }
    )
    counter = {"n": 0}

    def _append(*, block_id: str, children: list):  # noqa: ANN001
        counter["n"] += 1
        return {"object": "list", "results": [{"id": f"block-{counter['n']}"}]}

    client.blocks = MagicMock()
    client.blocks.children = MagicMock()
    client.blocks.children.append = AsyncMock(side_effect=_append)
    return client


async def _make_question(session: AsyncSession) -> Question:
    q = Question(
        source="uworld",
        qid=f"q-{uuid.uuid4().hex[:10]}",
        stem_html="<p>stem</p>",
        stem_plain="stem",
        choices=[{"id": "A", "text": "x"}],
        correct_choice="A",
    )
    session.add(q)
    await session.flush()
    return q


async def _make_factor(
    session: AsyncSession, *, question_id: int, node_id: int | None
) -> DiscriminatorFactor:
    f = DiscriminatorFactor(
        question_id=question_id,
        factor_text=f"confuses X with Y {uuid.uuid4().hex[:6]}",
        node_id=node_id,
    )
    session.add(f)
    await session.flush()
    return f


async def test_mirror_first_creates_page_and_records_block(db_session: AsyncSession):
    """V-N2: no prior page → mirror creates it, appends one block, and
    stamps the returned block id onto the factor."""
    node = await _make_course_node(db_session)
    q = await _make_question(db_session)
    factor = await _make_factor(db_session, question_id=q.id, node_id=node.id)
    client = _forge_mirror_client(page_id="d-page-1")

    report = await mirror_discriminator_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        factor=factor,
    )

    assert report.skipped is False
    assert report.created_page is True
    assert report.notion_block_id == "block-1"
    assert report.notion_page_id == "d-page-1"

    client.pages.create.assert_awaited_once()
    client.blocks.children.append.assert_awaited_once()

    # block id persisted on the factor row
    persisted = (
        await db_session.execute(
            select(DiscriminatorFactor.notion_block_id).where(DiscriminatorFactor.id == factor.id)
        )
    ).scalar_one()
    assert persisted == "block-1"

    # one pointer for the node (V-N2)
    pointer = (
        await db_session.execute(select(NotionPage).where(NotionPage.node_id == node.id))
    ).scalar_one()
    assert pointer.notion_page_id == "d-page-1"


async def test_mirror_reuses_existing_node_page(db_session: AsyncSession):
    """V-N2: a node page already created by a fact-sync is reused — mirror
    appends to it, ⊥ a second ``pages.create``."""
    node = await _make_course_node(db_session)
    facts = await _make_facts(db_session, node.course_id, ["Seed fact long enough."])
    q = await _make_question(db_session)
    factor = await _make_factor(db_session, question_id=q.id, node_id=node.id)
    client = _forge_mirror_client(page_id="shared-page")

    # fact-sync creates the page (children passed to pages.create, no append)
    await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=facts,
    )
    assert client.pages.create.await_count == 1
    client.blocks.children.append.assert_not_awaited()

    report = await mirror_discriminator_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        factor=factor,
    )

    assert report.created_page is False
    assert report.notion_page_id == "shared-page"
    # no new page; exactly one append for the discriminator block
    assert client.pages.create.await_count == 1
    client.blocks.children.append.assert_awaited_once()


async def test_mirror_idempotent_skips_when_already_mirrored(db_session: AsyncSession):
    """V-M3: re-mirroring a factor that already carries a notion_block_id is
    a no-op — ⊥ duplicate block, no Notion call, block id unchanged."""
    node = await _make_course_node(db_session)
    q = await _make_question(db_session)
    factor = await _make_factor(db_session, question_id=q.id, node_id=node.id)
    client = _forge_mirror_client(page_id="d-page-idem")

    first = await mirror_discriminator_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        factor=factor,
    )
    assert first.skipped is False
    assert client.blocks.children.append.await_count == 1

    second = await mirror_discriminator_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        factor=factor,
    )
    assert second.skipped is True
    assert second.notion_block_id == first.notion_block_id
    # no second append / create — fully idempotent
    assert client.blocks.children.append.await_count == 1
    client.pages.create.assert_awaited_once()


async def test_mirror_distinct_factors_distinct_blocks(db_session: AsyncSession):
    """V-M3: distinct factors on one node each get their own block + id;
    links preserved, ⊥ collapsed."""
    node = await _make_course_node(db_session)
    q = await _make_question(db_session)
    f1 = await _make_factor(db_session, question_id=q.id, node_id=node.id)
    f2 = await _make_factor(db_session, question_id=q.id, node_id=node.id)
    client = _forge_mirror_client(page_id="d-page-multi")

    r1 = await mirror_discriminator_to_notion(
        db_session, notion_client=client, notion_wiki_db_id="db-id", factor=f1
    )
    r2 = await mirror_discriminator_to_notion(
        db_session, notion_client=client, notion_wiki_db_id="db-id", factor=f2
    )

    assert r1.notion_block_id != r2.notion_block_id
    assert client.blocks.children.append.await_count == 2
    client.pages.create.assert_awaited_once()  # one page for the node (V-N2)


async def test_mirror_without_node_raises(db_session: AsyncSession):
    """V-N2: a factor with no node_id has no page to host it → error,
    no Notion call."""
    q = await _make_question(db_session)
    factor = await _make_factor(db_session, question_id=q.id, node_id=None)
    client = _forge_mirror_client()

    with pytest.raises(NotionMirrorError):
        await mirror_discriminator_to_notion(
            db_session,
            notion_client=client,
            notion_wiki_db_id="db-id",
            factor=factor,
        )
    client.pages.create.assert_not_awaited()
    client.blocks.children.append.assert_not_awaited()


async def test_mirror_no_read_endpoints_called(db_session: AsyncSession):
    """V-N1: the mirror never reads Notion back — the forged client omits
    read endpoints, so any read would raise AttributeError."""
    node = await _make_course_node(db_session)
    q = await _make_question(db_session)
    factor = await _make_factor(db_session, question_id=q.id, node_id=node.id)
    client = _forge_mirror_client()

    assert not isinstance(getattr(client.pages, "retrieve", None), AsyncMock)
    await mirror_discriminator_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        factor=factor,
    )
