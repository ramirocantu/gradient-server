"""RCA-33 live Notion smoke test — verifies the write-out seam actually
reaches Notion with the configured token + wiki DB.

Staged so each failure is diagnosable in isolation:
  1. users.me()                  — token valid + integration identity
  2. databases.retrieve(db_id)   — wiki DB reachable + integration shared in
  3. sync_node_to_notion(...)    — the real production path creates a page +
                                   appends fact bullets, persisting a pointer

Read-only stages (1,2) run first; the write stage (3) is gated behind --write
so connectivity can be checked without creating a Notion page.

Run:  uv run python scripts/notion_smoke.py            # stages 1-2 only
      uv run python scripts/notion_smoke.py --write     # + real page create
"""

from __future__ import annotations

import asyncio
import sys

from app.config import settings


async def main(do_write: bool, keep: bool) -> int:
    if not settings.NOTION_API_TOKEN or not settings.NOTION_WIKI_DB_ID:
        print("FAIL: NOTION_API_TOKEN / NOTION_WIKI_DB_ID unset")
        return 1

    from notion_client import AsyncClient

    client = AsyncClient(auth=settings.NOTION_API_TOKEN)
    try:
        # Stage 1 — token valid.
        me = await client.users.me()
        bot_name = me.get("name") or me.get("bot", {}).get("owner", {}).get("type", "?")
        print(f"[1] users.me OK — integration='{bot_name}' id={me.get('id')}")

        # Stage 2 — wiki data source reachable + shared with integration.
        # API 2025-09-03: the configured id is a data_source id, retrieved via
        # the data_sources endpoint (databases.retrieve expects a database id).
        ds = await client.request(
            path=f"data_sources/{settings.NOTION_WIKI_DB_ID}", method="GET"
        )
        title = "".join(t.get("plain_text", "") for t in ds.get("title", []))
        props = list(ds.get("properties", {}).keys())
        print(f"[2] data_sources.retrieve OK — title='{title}' props={props}")
        if "Name" not in ds.get("properties", {}):
            print("    WARN: data source has no 'Name' title property; pages.create sets 'Name'")

        if not do_write:
            print("[3] skipped (pass --write to create a real page)")
            return 0

        # Stage 3 — real production write path.
        from app.database import AsyncSessionLocal
        from app.models.outline import OutlineNode
        from app.models.atomic_fact import AtomicFact
        from app.services.kb.notion import sync_node_to_notion
        from sqlalchemy import select

        from app.models.notion_page import NotionPage

        async with AsyncSessionLocal() as session:
            node = (
                await session.execute(select(OutlineNode).limit(1))
            ).scalar_one_or_none()
            if node is None:
                print("[3] FAIL: no outline node in this branch DB to host a page")
                return 1
            # Repeatable: drop any stale pointer from a prior smoke run so this
            # always exercises the first-sync pages.create path, not an append
            # to an already-archived page.
            stale = (
                await session.execute(select(NotionPage).where(NotionPage.node_id == node.id))
            ).scalar_one_or_none()
            if stale is not None:
                await session.delete(stale)
                await session.flush()
            # Synthetic in-memory fact (not persisted) just to exercise blocks.
            fact = AtomicFact(text="RCA-33 smoke fact — Notion write path live check")
            report = await sync_node_to_notion(
                session,
                notion_client=client,
                notion_wiki_db_id=settings.NOTION_WIKI_DB_ID,
                node=node,
                facts=[fact],
            )
            page_id = report.notion_page_id
            print(
                f"[3] sync_node_to_notion OK — node='{node.name}' "
                f"page_id={page_id} created={report.created_page} "
                f"blocks={report.appended_blocks}"
            )
            if keep:
                await session.commit()
                print(f"[3] page KEPT (--keep) — view: {report and report.notion_page_id}")
                print(f"    url: https://www.notion.so/{page_id.replace('-', '')}")
            else:
                # Cleanup: archive the proof page AND drop the pointer row so the
                # live wiki isn't littered and re-runs start clean.
                await client.pages.update(page_id=page_id, archived=True)
                pointer = (
                    await session.execute(
                        select(NotionPage).where(NotionPage.node_id == node.id)
                    )
                ).scalar_one_or_none()
                if pointer is not None:
                    await session.delete(pointer)
                await session.commit()
                print("[3] proof page archived + pointer row dropped (cleanup)")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        await client.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main("--write" in sys.argv, "--keep" in sys.argv)))
