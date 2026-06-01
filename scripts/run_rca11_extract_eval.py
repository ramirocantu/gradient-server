"""RCA-11 extraction-quality measurement harness (V-L2).

Measures whether the document-level learning-goals rework (pdf-vision-v2) cuts
lecture-context noise without gutting genuine fact yield. Two modes:

  # 1. Capture the reference transcriptions ONCE (one real vision pass per page)
  python -m scripts.run_rca11_extract_eval \\
      --pdf "/Users/rcantu/Desktop/Phys McatKing.pdf" \\
      --save-fixture tests/fixtures/rca11_reference_transcriptions.json

  # 2. Replay extraction over saved transcriptions (cheap, deterministic — no
  #    vision, no PyMuPDF, no DB; isolates the one changed variable)
  python -m scripts.run_rca11_extract_eval \\
      --fixture tests/fixtures/rca11_reference_transcriptions.json \\
      --report data/rca11_extract_report.json

Both modes run document-level extraction (concat → chunk → extract per chunk),
print yield / kept / dropped-context / keyword-noise-fraction, and DUMP the kept
facts to stdout for the human "memorizable?" judgment — the gate stays
human-judged (cognitive-safety: AI is not the arbiter of recall-worthiness).

Lives outside `app/` so it never imports into production paths.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.eval.extract_metrics import summarize
from app.services.kb import pdf_ingest
from app.services.llm.client import build_openai_client

logger = logging.getLogger("rca11_extract_eval")


async def _capture_transcriptions(pdf_path: Path, *, client: Any) -> list[dict[str, Any]]:
    """One real vision pass per page → `[{page, text}]` (the saved fixture)."""
    service_tier = settings.OPENAI_SERVICE_TIER
    model = settings.OPENAI_VISION_MODEL or settings.OPENAI_MODEL
    pages = pdf_ingest.render_pages(pdf_path)
    out: list[dict[str, Any]] = []
    for rendered in pages:
        t = await pdf_ingest.transcribe_page(
            rendered.image_png, client=client, model=model, service_tier=service_tier
        )
        out.append({"page": rendered.page, "text": t.text})
        logger.info("transcribed page %d (%d chars)", rendered.page, len(t.text))
    return out


def _load_transcriptions(path: Path) -> list[tuple[int, str]]:
    raw = json.loads(path.read_text())
    return [(int(r["page"]), str(r["text"])) for r in raw if str(r.get("text", "")).strip()]


async def _extract_document(
    transcribed: list[tuple[int, str]], *, client: Any
) -> list[tuple[str, str]]:
    """Document-level extraction over the same chunking the ingest path uses."""
    service_tier = settings.OPENAI_SERVICE_TIER
    model = settings.OPENAI_MODEL
    facts: list[tuple[str, str]] = []
    chunks = pdf_ingest._chunk_document(transcribed)
    for lead_page, chunk_text in chunks:
        extraction = await pdf_ingest.extract_atomic_facts(
            chunk_text, client=client, model=model, service_tier=service_tier
        )
        logger.info("extracted %d facts from chunk lead-page %d", len(extraction.facts), lead_page)
        facts.extend(extraction.facts)
    return facts


async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = build_openai_client(max_retries=5)

    if args.pdf:
        transcriptions = await _capture_transcriptions(Path(args.pdf), client=client)
        if args.save_fixture:
            out = Path(args.save_fixture)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(transcriptions, indent=2) + "\n", encoding="utf-8")
            print(f"saved {len(transcriptions)} page transcriptions -> {out}")
        transcribed = [
            (int(r["page"]), str(r["text"]))
            for r in transcriptions
            if str(r.get("text", "")).strip()
        ]
    elif args.fixture:
        transcribed = _load_transcriptions(Path(args.fixture))
    else:
        raise SystemExit("pass --pdf (capture) or --fixture (replay)")

    if not transcribed:
        raise SystemExit("no non-empty transcriptions to extract from")

    facts = await _extract_document(transcribed, client=client)
    result = summarize(settings.OPENAI_MODEL, facts)

    print(
        f"\n=== RCA-11 extract eval ({pdf_ingest.EXTRACTOR_VERSION}, "
        f"model={settings.OPENAI_MODEL}) ===\n"
        f"pages={len(transcribed)}  total_facts={result.total}  "
        f"kept(concept)={result.kept}  dropped(context)={result.dropped_context}  "
        f"keyword_noise_fraction(kept)={result.noise_fraction:.3f}\n"
    )
    print("--- kept facts (human 'memorizable?' judgment) ---")
    for text, kind in facts:
        if kind == "concept":
            print(f"  • {text}")

    if args.report:
        rep = Path(args.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "extractor_version": pdf_ingest.EXTRACTOR_VERSION,
            "result": result.as_dict(),
            "kept_facts": [t for t, k in facts if k == "concept"],
            "dropped_facts": [t for t, k in facts if k != "concept"],
        }
        rep.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nreport -> {rep}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RCA-11 extraction-quality harness (V-L2)")
    parser.add_argument("--pdf", help="capture mode: PDF to vision-transcribe once")
    parser.add_argument("--save-fixture", help="where to write captured transcriptions (with --pdf)")
    parser.add_argument("--fixture", help="replay mode: saved transcriptions JSON")
    parser.add_argument("--report", help="optional JSON report path")
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
