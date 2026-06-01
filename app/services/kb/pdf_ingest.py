"""PDF ingest — vision transcription + grounded atomic-fact extraction (T54).

Notes-ingress redesign (2026-05-28): lecture notes / slidedecks arrive as
PDFs that frequently have **no extractable text** — handwriting, scanned
pages, image-only slides. So we no longer trust ``pdfplumber.extract_text``.
Instead every page is rendered to an image (PyMuPDF) and transcribed by an
OpenAI **vision** call (V-KB3); the transcription is then handed to an OpenAI
structured-output call that emits atomic factual claims (V-KB4). Facts persist
to ``atomic_facts`` with ``node_id`` NULL — the grounded-tag categorizer
(V-L3/V69, T50) assigns the node later.

Idempotent (V-KB1): re-ingesting a file with the same SHA-256 returns the
existing ``pdf_sources`` row and writes no new facts. ``UQ(course_id,
content_hash)`` on ``atomic_facts`` is the second line of defense.

Both the page renderer and the OpenAI clients are injected so tests never
render a real PDF or hit the API (V16): the renderer default uses PyMuPDF;
the vision + extraction clients are mocked at the SDK boundary.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.atomic_fact import AtomicFact
from app.models.pdf_source import PdfSource

_logger = logging.getLogger("app.services.kb.pdf_ingest")

# Bump when the vision prompt / extraction schema changes meaningfully.
# Stamped onto every persisted fact (V-KB3) so a re-run under a new version
# is a clean miss once content_hash dedup is keyed differently downstream.
# v2 (RCA-11): document-level learning-goals synthesis + concept/context
# discernment gate + enriched vision (figure semantics).
EXTRACTOR_VERSION = "pdf-vision-v2"

_RENDER_DPI = 150
_VISION_MAX_TOKENS = 4096
# Doc-level extraction (V-KB5): one call now covers the whole document, so the
# output budget is larger than the old per-page cap.
_EXTRACT_MAX_TOKENS = 3072
# Concat-doc char ceiling before we split extraction on page boundaries (V-KB5).
# Keeps a multi-page document inside nano's context with room for the 3072 output.
_EXTRACT_MAX_DOC_CHARS = 48_000

_VISION_SYSTEM = (
    "You transcribe a single page from a student's lecture notes or slide deck. "
    "The page may be typed, a slide image, or handwritten. Output the full "
    "readable text content of the page, faithfully and verbatim where legible. "
    "Transcribe handwriting as best you can.\n"
    "In ADDITION to the text, when the page contains a diagram, figure, graph, "
    "chart, or table whose meaning is not fully captured by its text, emit a "
    "brief factual description of it on its own line, prefixed `[figure]` "
    "(e.g. `[figure] free-body diagram: block on an incline, friction vector "
    "down-slope, gravity decomposed into components`). Describe only what is "
    "shown — labels, relationships, quantities, structure. Do NOT add outside "
    "knowledge, commentary, or interpretation beyond the visual. If the page has "
    "no legible text and no informative figure, output nothing."
)

_EXTRACT_SYSTEM = (
    "You are given the full transcribed text of ONE study document (lecture notes "
    "or a slide deck), with `=== page N ===` markers between pages. Produce the set "
    "of atomic declarative facts a student is expected to LEARN from it.\n"
    "\n"
    "First, internally consider the document's learning goals: what key concepts, "
    "definitions, principles, mechanisms, relationships, and quantitative laws "
    "should a student understand after studying it? Then emit atomic facts that "
    "COVER those goals.\n"
    "\n"
    "Each fact must be ONE self-contained declarative statement, true on its own "
    "without the surrounding text. You MAY synthesize or restate a fact into clean "
    "standalone form even when the document never phrases it as a single sentence "
    "(summarization is allowed) — but stay grounded in the document's content and "
    "do NOT introduce outside facts.\n"
    "\n"
    "Classify each fact with `kind`:\n"
    "- `concept` — a durable, study-worthy declarative fact (definition, principle, "
    "mechanism, relationship, quantitative law). This is the ONLY kind that is kept.\n"
    "- `context` — describes a worked example, an example/practice-question setup, "
    "what a problem asks, or one specific figure/diagram instance. Mark these "
    "`context`, NOT `concept`.\n"
    "\n"
    "Do NOT emit at all (not even as context): slide titles, page numbers, headers, "
    "author/course/date lines, and meta-statements about \"the text\", \"this page\", "
    "\"the slide\", \"the diagram\", or \"the example below\".\n"
    "\n"
    "You produce declarative statements only. You must NEVER write a question, a "
    "prompt, a flashcard, or any active-recall item — statements of fact only."
)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class RenderedPage:
    page: int
    image_png: bytes


@dataclass
class PageTranscription:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


# A parsed fact: its text plus its discernment kind ('concept' kept, 'context'
# dropped at persist per V-KB7).
ParsedFact = tuple[str, str]


@dataclass
class FactExtraction:
    facts: list[ParsedFact] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class IngestReport:
    pdf_source_id: int
    new_facts: int
    dup_facts: int
    pages: int
    reused_pdf: bool
    extractor_version: str = EXTRACTOR_VERSION
    # V-KB7: example/lecture-context facts the model flagged 'context' and we
    # dropped before persist.
    dropped_context_facts: int = 0
    # V-L1: token accounting summed across every vision + extraction call.
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


# Injection seam types.
Renderer = Callable[[Path], list[RenderedPage]]


# --------------------------------------------------------------------------- #
# Page render (PyMuPDF) — injectable
# --------------------------------------------------------------------------- #


def render_pages(path: Path, *, dpi: int = _RENDER_DPI) -> list[RenderedPage]:
    """Default renderer: rasterize each PDF page to a PNG via PyMuPDF.

    Heavy import deferred to call time so the module imports cheaply and tests
    that inject a forged renderer never load PyMuPDF.
    """

    import pymupdf  # noqa: PLC0415 — heavy import only when rendering for real

    pages: list[RenderedPage] = []
    with pymupdf.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):  # pyright: ignore[reportArgumentType]  — pymupdf has no stubs
            pix = page.get_pixmap(dpi=dpi)
            pages.append(RenderedPage(page=i, image_png=pix.tobytes("png")))
    return pages


# --------------------------------------------------------------------------- #
# Usage accounting (V-L1)
# --------------------------------------------------------------------------- #


def _read_usage(completion: Any) -> tuple[int, int, int]:
    """Return ``(prompt_tokens, output_tokens, cached_tokens)`` from a
    ChatCompletion. Cache hits come from ``prompt_tokens_details.cached_tokens``
    — never inferred (V-L1)."""

    usage = getattr(completion, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cached_tokens = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
    return prompt_tokens, output_tokens, cached_tokens


def _message_content(completion: Any) -> str | None:
    choices = getattr(completion, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None) if choice is not None else None
    content = getattr(message, "content", None) if message is not None else None
    return content


# --------------------------------------------------------------------------- #
# Vision transcription (V-KB3)
# --------------------------------------------------------------------------- #


async def transcribe_page(
    image_png: bytes,
    *,
    client: Any,
    model: str,
    max_tokens: int = _VISION_MAX_TOKENS,
    service_tier: str | None = None,
) -> PageTranscription:
    """One OpenAI vision call: page image → transcribed text (V-KB3).

    ``client`` is an ``AsyncOpenAI``-shaped object, injected + mocked at the
    SDK boundary in tests (V16). ``service_tier`` (e.g. ``'flex'``, V-L5) is
    forwarded when set."""

    b64 = base64.b64encode(image_png).decode("ascii")
    create_kwargs: dict[str, Any] = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": _VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this page."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
    }
    if service_tier is not None:
        create_kwargs["service_tier"] = service_tier
    completion = await client.chat.completions.create(**create_kwargs)
    prompt_tokens, output_tokens, cached_tokens = _read_usage(completion)
    text = (_message_content(completion) or "").strip()
    return PageTranscription(
        text=text,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )


# --------------------------------------------------------------------------- #
# Atomic-fact extraction (V-KB4, V45 structured output)
# --------------------------------------------------------------------------- #


_EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "extract_atomic_facts",
        "description": "Atomic declarative facts grounded in the document text.",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["facts"],
            "additionalProperties": False,
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["text", "kind"],
                        "additionalProperties": False,
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "One self-contained atomic declarative claim.",
                            },
                            "kind": {
                                "type": "string",
                                "enum": ["concept", "context"],
                                "description": (
                                    "concept = durable study-worthy fact (kept); "
                                    "context = example/figure/problem prose (dropped)."
                                ),
                            },
                        },
                    },
                },
            },
        },
    },
}


def _parse_facts(payload: dict[str, Any]) -> list[ParsedFact]:
    out: list[ParsedFact] = []
    for raw in payload.get("facts") or []:
        if not isinstance(raw, dict):
            continue
        text = raw.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        # V-KB7: default to 'concept' on a missing/invalid kind — never silently
        # drop a fact because the discriminator field was malformed.
        kind = raw.get("kind")
        if kind not in ("concept", "context"):
            kind = "concept"
        out.append((text, kind))
    return out


async def extract_atomic_facts(
    text: str,
    *,
    client: Any,
    model: str,
    max_tokens: int = _EXTRACT_MAX_TOKENS,
    service_tier: str | None = None,
) -> FactExtraction:
    """One OpenAI structured-output call: document text → atomic facts (V-KB4).

    ``text`` is the full document (page transcriptions concatenated, V-KB5), not a
    single page. Empty/blank input → no LLM call, empty result. Strict json_schema
    emits the document in ``choice.message.content`` (mirrors ``llm/grounded.py``).
    ``service_tier`` (e.g. ``'flex'``, V-L5) is forwarded when set."""

    if not text.strip():
        return FactExtraction()

    create_kwargs: dict[str, Any] = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": text.strip()},
        ],
        "response_format": _EXTRACT_SCHEMA,
    }
    if service_tier is not None:
        create_kwargs["service_tier"] = service_tier
    completion = await client.chat.completions.create(**create_kwargs)
    prompt_tokens, output_tokens, cached_tokens = _read_usage(completion)

    content = _message_content(completion)
    facts: list[ParsedFact] = []
    if content:
        try:
            facts = _parse_facts(json.loads(content))
        except json.JSONDecodeError as exc:
            _logger.warning("extract: response content not valid JSON: %s", exc)
    return FactExtraction(
        facts=facts,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _format_page(page: int, text: str) -> str:
    return f"=== page {page} ===\n{text}"


def _chunk_document(
    transcribed: list[tuple[int, str]],
    *,
    max_chars: int = _EXTRACT_MAX_DOC_CHARS,
) -> list[tuple[int, str]]:
    """Group page transcriptions into document chunks for extraction (V-KB5).

    Pages are concatenated with ``=== page N ===`` markers into as few chunks as
    fit under ``max_chars``; splits fall only on page boundaries. Each chunk is
    returned with its lead (first) page, used to stamp ``AtomicFact.page``. A
    single page longer than ``max_chars`` becomes its own chunk (never split
    mid-page). Empty input → no chunks (no extraction call)."""

    chunks: list[tuple[int, str]] = []
    cur: list[str] = []
    cur_lead = 0
    cur_len = 0
    for page, text in transcribed:
        block = _format_page(page, text)
        sep = 2 if cur else 0  # cost of the '\n\n' join
        if cur and cur_len + sep + len(block) > max_chars:
            chunks.append((cur_lead, "\n\n".join(cur)))
            cur, cur_lead, cur_len, sep = [], 0, 0, 0
        if not cur:
            cur_lead = page
        cur.append(block)
        cur_len += sep + len(block)
    if cur:
        chunks.append((cur_lead, "\n\n".join(cur)))
    return chunks


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def ingest_pdf(
    session: AsyncSession,
    *,
    course_id: int,
    path: Path,
    vision_client: Any,
    extract_client: Any | None = None,
    renderer: Renderer = render_pages,
    vision_model: str | None = None,
    extract_model: str | None = None,
    extractor_version: str = EXTRACTOR_VERSION,
) -> IngestReport:
    """Render → vision-transcribe → extract facts → persist (V-KB3, V-KB4).

    Args:
        course_id: owning course (atomic_facts dedup scope).
        path: the PDF on disk.
        vision_client: injected ``AsyncOpenAI`` for page transcription (V16).
        extract_client: client for fact extraction; defaults to ``vision_client``.
        renderer: page→image renderer; defaults to PyMuPDF, injectable for tests.
        vision_model / extract_model: default to ``OPENAI_VISION_MODEL`` (falling
            back to ``OPENAI_MODEL``) / ``OPENAI_MODEL``.

    Runs inside the caller's transaction — the caller owns commit/rollback.
    """

    extract_client = extract_client or vision_client
    resolved_vision_model = vision_model or settings.OPENAI_VISION_MODEL or settings.OPENAI_MODEL
    resolved_extract_model = extract_model or settings.OPENAI_MODEL
    service_tier = settings.OPENAI_SERVICE_TIER  # V-L5 Flex (None omits)

    sha = _sha256_file(path)

    existing = (
        await session.execute(select(PdfSource).where(PdfSource.sha256 == sha))
    ).scalar_one_or_none()
    if existing is not None:
        return IngestReport(
            pdf_source_id=existing.id,
            new_facts=0,
            dup_facts=0,
            pages=0,
            reused_pdf=True,
            extractor_version=extractor_version,
        )

    pdf_row = PdfSource(
        course_id=course_id,
        filename=path.name,
        sha256=sha,
        status="parsing",
    )
    session.add(pdf_row)
    await session.flush()
    pdf_id = pdf_row.id

    pages = renderer(path)
    new_facts = 0
    dup_facts = 0
    dropped_context = 0
    in_tokens = 0
    out_tokens = 0
    cached = 0
    seen_hashes: set[str] = set()

    # Pass 1 — transcribe every page (vision is inherently per-image, V-KB8).
    transcribed: list[tuple[int, str]] = []
    for rendered in pages:
        transcription = await transcribe_page(
            rendered.image_png,
            client=vision_client,
            model=resolved_vision_model,
            service_tier=service_tier,
        )
        in_tokens += transcription.prompt_tokens
        out_tokens += transcription.output_tokens
        cached += transcription.cached_tokens
        if transcription.text:
            transcribed.append((rendered.page, transcription.text))

    # Pass 2 — document-level extraction (V-KB5): concat page transcriptions and
    # run one extraction per chunk so the model sees the whole learning arc.
    for chunk_lead_page, chunk_text in _chunk_document(transcribed):
        extraction = await extract_atomic_facts(
            chunk_text,
            client=extract_client,
            model=resolved_extract_model,
            service_tier=service_tier,
        )
        in_tokens += extraction.prompt_tokens
        out_tokens += extraction.output_tokens
        cached += extraction.cached_tokens

        for fact_text, kind in extraction.facts:
            # V-KB7: discernment gate — only durable 'concept' facts persist;
            # 'context' (example/figure/problem prose) is dropped and counted.
            if kind != "concept":
                dropped_context += 1
                continue
            content_hash = _sha256_text(fact_text)
            if content_hash in seen_hashes:
                dup_facts += 1
                continue
            existing_fact = (
                await session.execute(
                    select(AtomicFact).where(
                        AtomicFact.course_id == course_id,
                        AtomicFact.content_hash == content_hash,
                    )
                )
            ).scalar_one_or_none()
            if existing_fact is not None:
                seen_hashes.add(content_hash)
                dup_facts += 1
                continue
            session.add(
                AtomicFact(
                    course_id=course_id,
                    pdf_source_id=pdf_id,
                    page=chunk_lead_page,
                    text=fact_text,
                    content_hash=content_hash,
                    extractor_version=extractor_version,
                )
            )
            seen_hashes.add(content_hash)
            new_facts += 1

    pdf_row.status = "ingested"
    pdf_row.ingested_at = datetime.now(timezone.utc)
    await session.flush()

    _logger.info(
        "pdf_ingest: pdf=%d pages=%d new_facts=%d dup_facts=%d dropped_context=%d "
        "vision_model=%s extract_model=%s prompt=%d cached=%d out=%d version=%s",
        pdf_id,
        len(pages),
        new_facts,
        dup_facts,
        dropped_context,
        resolved_vision_model,
        resolved_extract_model,
        in_tokens,
        cached,
        out_tokens,
        extractor_version,
    )

    return IngestReport(
        pdf_source_id=pdf_id,
        new_facts=new_facts,
        dup_facts=dup_facts,
        pages=len(pages),
        reused_pdf=False,
        extractor_version=extractor_version,
        dropped_context_facts=dropped_context,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        cached_tokens=cached,
    )
