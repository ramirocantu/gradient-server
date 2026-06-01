"""Notion write-out seam (T26 + T32, V-N1, V-N2, V-M3, V16).

One-way mirror: Postgres → Notion. Never reads Notion content back
(V-N1); the only stored Notion state is the ``notion_pages`` pointer
(page_id, url, tags snapshot, node_id) used for backlinks. One Notion
page per outline node (V-N2) — re-sync upserts on ``node_id``.

T32 adds the discriminator-factor mirror: a persisted
``DiscriminatorFactor`` (T31) is appended as one block on its node's
Notion page with a one-way backlink anchor to the source question +
node, and its ``notion_block_id`` is recorded. Idempotent (V-M3): a
factor already carrying a ``notion_block_id`` is a no-op — ⊥ duplicate
block, ⊥ page rewrite.

The notion-client ``AsyncClient`` is injected so tests mock at the
SDK boundary (V16). All write operations only — ``pages.create``,
``blocks.children.append``; we never call read endpoints
(``pages.retrieve`` / ``databases.query``) here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.captures import Question
from app.models.discriminator_factor import DiscriminatorFactor
from app.models.notion_page import NotionPage
from app.models.outline import OutlineNode

_logger = logging.getLogger("app.services.kb.notion")


class NotionMirrorError(RuntimeError):
    """Raised when a discriminator factor cannot be mirrored to Notion
    (no ``node_id`` to host the page per V-N2, a dangling FK, or an append
    that returned no block id)."""


@dataclass
class SyncReport:
    notion_page_row_id: int
    notion_page_id: str
    created_page: bool  # True on first sync, False on subsequent
    appended_blocks: int


@dataclass
class DiscriminatorSyncReport:
    factor_id: int
    notion_page_id: str
    notion_block_id: str
    created_page: bool  # True when the node page was created by this call
    skipped: bool  # True when already mirrored (V-M3 idempotent no-op)


def _fact_blocks(facts: Iterable[AtomicFact]) -> list[dict[str, Any]]:
    """Render atomic facts as Notion bulleted_list_item blocks. Append-only
    on re-sync (V-M3: ⊥ rewrite). Each fact's text becomes a single
    bullet — chunking finer than the sentence is the LLM4Tag job (T29),
    not this seam.
    """

    blocks: list[dict[str, Any]] = []
    for f in facts:
        blocks.append(
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f.text}}],
                },
            }
        )
    return blocks


def _tag_snapshot(facts: Iterable[AtomicFact], *, prev: list[Any] | None) -> list[str]:
    snapshot = list(prev or [])
    for f in facts:
        head = f.text[:50]
        if head not in snapshot:
            snapshot.append(head)
    return snapshot


async def _ensure_node_page(
    session: AsyncSession,
    *,
    notion_client: Any,
    notion_wiki_db_id: str,
    node: OutlineNode,
    initial_blocks: list[dict[str, Any]],
) -> tuple[NotionPage, bool]:
    """Return the ``notion_pages`` pointer for ``node``, creating the page
    on first call (V-N2: one page per node, keyed by ``node_id`` UQ).

    Returns ``(pointer, created)``. When a pointer already exists it is
    returned untouched with ``created=False`` and **no** Notion call —
    re-sync upserts, ⊥ duplicates (V-N2). On create, ``pages.create`` runs
    with ``initial_blocks`` as the page's first children; the caller owns
    any ``tags`` / ``last_synced_at`` bookkeeping it needs afterward.
    """

    pointer = (
        await session.execute(select(NotionPage).where(NotionPage.node_id == node.id))
    ).scalar_one_or_none()
    if pointer is not None:
        return pointer, False

    result = await notion_client.pages.create(
        # Notion API 2025-09-03 (notion-client>=3): a database's rows live under
        # a *data source*, so page creation parents to the data_source_id, not
        # the legacy database_id (which 404s against a data-source id). The
        # configured NOTION_WIKI_DB_ID is the data-source id.
        parent={"type": "data_source_id", "data_source_id": notion_wiki_db_id},
        properties={
            "Name": {"title": [{"type": "text", "text": {"content": node.name}}]},
        },
        children=initial_blocks,
    )
    pointer = NotionPage(
        node_id=node.id,
        notion_page_id=result["id"],
        url=result.get("url", ""),
        tags=None,
        last_synced_at=datetime.now(timezone.utc),
    )
    session.add(pointer)
    await session.flush()
    return pointer, True


async def sync_node_to_notion(
    session: AsyncSession,
    *,
    notion_client: Any,
    notion_wiki_db_id: str,
    node: OutlineNode,
    facts: list[AtomicFact],
) -> SyncReport:
    """Upsert a Notion page for ``node`` + append fact blocks.

    First sync (no ``notion_pages`` row yet): ``pages.create`` with the
    initial bulleted facts; persist a pointer row with page id/url.
    Subsequent sync: ``blocks.children.append`` adds new fact blocks
    underneath the same page — V-M3 append-only, V-N1 ⊥ rewrite.

    Caller owns dedupe: ``facts`` should be the slice the caller wants
    appended this run; passing the full set on every re-sync would
    duplicate blocks. (Atomic-fact dedupe lives one layer up, on the
    Postgres side via ``UQ(course_id, content_hash)``.)
    """

    blocks = _fact_blocks(facts)

    pointer, created = await _ensure_node_page(
        session,
        notion_client=notion_client,
        notion_wiki_db_id=notion_wiki_db_id,
        node=node,
        initial_blocks=blocks,
    )

    if created:
        pointer.tags = _tag_snapshot(facts, prev=None)
        pointer.last_synced_at = datetime.now(timezone.utc)
        await session.flush()
        return SyncReport(
            notion_page_row_id=pointer.id,
            notion_page_id=pointer.notion_page_id,
            created_page=True,
            appended_blocks=len(blocks),
        )

    if blocks:
        await notion_client.blocks.children.append(
            block_id=pointer.notion_page_id,
            children=blocks,
        )

    pointer.tags = _tag_snapshot(facts, prev=pointer.tags)
    pointer.last_synced_at = datetime.now(timezone.utc)
    await session.flush()
    return SyncReport(
        notion_page_row_id=pointer.id,
        notion_page_id=pointer.notion_page_id,
        created_page=False,
        appended_blocks=len(blocks),
    )


def _discriminator_block(
    factor: DiscriminatorFactor, *, question: Question, node: OutlineNode
) -> dict[str, Any]:
    """Render a discriminator factor as one Notion block with a one-way
    backlink anchor (V-N1).

    The block is the factor text followed by a gray inline reference to the
    source question + node — an *anchor* identifying provenance, ⊥ a Notion
    relation/read-back. The node page itself is the node anchor (V-N2); the
    ``Q:<qid>`` fragment is the question backlink.
    """

    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [
                {"type": "text", "text": {"content": factor.factor_text}},
                {
                    "type": "text",
                    "text": {"content": f"  ↪ Q:{question.qid} · {node.name}"},
                    "annotations": {"italic": True, "color": "gray"},
                },
            ],
        },
    }


async def mirror_discriminator_to_notion(
    session: AsyncSession,
    *,
    notion_client: Any,
    notion_wiki_db_id: str,
    factor: DiscriminatorFactor,
) -> DiscriminatorSyncReport:
    """Append a persisted discriminator factor (T31) to its node's Notion
    page as one backlink block, and record the ``notion_block_id`` (T32).

    Idempotent (V-M3): a factor already carrying a ``notion_block_id`` is a
    no-op — returns ``skipped=True`` with **no** Notion call, ⊥ a duplicate
    block. Otherwise: ensure the node's page exists (creating it if no
    fact-sync has run yet — V-N2), ``blocks.children.append`` exactly one
    block (append-only, ⊥ page rewrite — V-N1), capture the new block id
    from the append result, and stamp it onto ``factor.notion_block_id``.

    Raises:
        NotionMirrorError: ``factor.node_id`` is None (V-N2 needs a node to
            host the page), the node/question FK is dangling, or the append
            returned no block id.
    """

    # V-M3 idempotent: already mirrored → no-op, ⊥ duplicate block.
    if factor.notion_block_id is not None:
        return DiscriminatorSyncReport(
            factor_id=factor.id,
            notion_page_id="",
            notion_block_id=factor.notion_block_id,
            created_page=False,
            skipped=True,
        )

    if factor.node_id is None:
        raise NotionMirrorError(
            f"discriminator factor id={factor.id} has no node_id; "
            "V-N2 requires a node to host the Notion page"
        )

    node = (
        await session.execute(select(OutlineNode).where(OutlineNode.id == factor.node_id))
    ).scalar_one_or_none()
    if node is None:
        raise NotionMirrorError(f"node id={factor.node_id} not found")

    question = (
        await session.execute(select(Question).where(Question.id == factor.question_id))
    ).scalar_one_or_none()
    if question is None:
        raise NotionMirrorError(f"question id={factor.question_id} not found")

    pointer, created = await _ensure_node_page(
        session,
        notion_client=notion_client,
        notion_wiki_db_id=notion_wiki_db_id,
        node=node,
        initial_blocks=[],
    )

    block = _discriminator_block(factor, question=question, node=node)
    result = await notion_client.blocks.children.append(
        block_id=pointer.notion_page_id,
        children=[block],
    )
    try:
        block_id = result["results"][0]["id"]
    except (KeyError, IndexError, TypeError) as exc:
        raise NotionMirrorError(
            "Notion append returned no block id; cannot record notion_block_id"
        ) from exc

    factor.notion_block_id = block_id
    pointer.last_synced_at = datetime.now(timezone.utc)
    await session.flush()

    _logger.info(
        "mirror_discriminator_to_notion: factor_id=%d node_id=%d block_id=%s created_page=%s",
        factor.id,
        node.id,
        block_id,
        created,
    )
    return DiscriminatorSyncReport(
        factor_id=factor.id,
        notion_page_id=pointer.notion_page_id,
        notion_block_id=block_id,
        created_page=created,
        skipped=False,
    )


@dataclass
class PendingSyncReport:
    nodes_synced: int = 0
    pages_created: int = 0
    blocks_appended: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def partial_failure(self) -> bool:
        return bool(self.failures)


async def sync_pending_nodes(
    session: AsyncSession,
    *,
    notion_client: Any,
    notion_wiki_db_id: str,
) -> PendingSyncReport:
    """Mirror every node that owns tagged atomic facts to Notion (T51).

    A fact is mirrorable only once the grounded-tag categorizer (T50) has set
    its ``node_id`` — so this is empty until tagging runs. For each such node:
    create the page on first sync (all its facts as initial blocks), else
    append only facts created since ``notion_pages.last_synced_at`` — the
    append-only, ⊥-rewrite discipline (V-N1) that keeps re-sync idempotent
    (V-N2: one page per node). Per-node failures are caught (V41) so one bad
    node ⊥ aborts the batch; the caller still reaches ``commit()``.
    """

    node_ids = (
        (
            await session.execute(
                select(AtomicFact.node_id).where(AtomicFact.node_id.is_not(None)).distinct()
            )
        )
        .scalars()
        .all()
    )

    report = PendingSyncReport()
    for nid in node_ids:
        try:
            node = (
                await session.execute(select(OutlineNode).where(OutlineNode.id == nid))
            ).scalar_one_or_none()
            if node is None:
                continue
            pointer = (
                await session.execute(select(NotionPage).where(NotionPage.node_id == nid))
            ).scalar_one_or_none()

            stmt = select(AtomicFact).where(AtomicFact.node_id == nid)
            if pointer is not None:
                # Append-only: only facts newer than the last sync (V-N1).
                stmt = stmt.where(AtomicFact.created_at > pointer.last_synced_at)
            facts = (await session.execute(stmt.order_by(AtomicFact.id))).scalars().all()

            if not facts and pointer is not None:
                continue  # nothing new for this node

            sync_report = await sync_node_to_notion(
                session,
                notion_client=notion_client,
                notion_wiki_db_id=notion_wiki_db_id,
                node=node,
                facts=list(facts),
            )
            report.nodes_synced += 1
            if sync_report.created_page:
                report.pages_created += 1
            report.blocks_appended += sync_report.appended_blocks
        except Exception as exc:  # noqa: BLE001 — V41 per-node isolation
            report.failures.append(f"node {nid}: {exc}")
            _logger.warning("sync_pending_nodes: failed on node %s: %s", nid, exc)

    _logger.info(
        "sync_pending_nodes: nodes=%d pages_created=%d blocks=%d failures=%d",
        report.nodes_synced,
        report.pages_created,
        report.blocks_appended,
        len(report.failures),
    )
    return report
