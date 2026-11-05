"""LLM clause extraction with page citations and a programmatic citation check.

The LLM (Google Gemini) only proposes clauses. Every proposal carries a page number
and a verbatim quote; check_quote() then looks for that quote in the PDF text so a
human can see at a glance which extractions are grounded and which may be invented.
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone

from pypdf import PdfReader
from rapidfuzz import fuzz

DEFAULT_MODEL = "gemini-3.8-flash"
MODEL_CHOICES = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
MAX_CHARS = 300_000  # roughly 75k tokens; larger documents are pre-filtered by keyword

# What we ask the model for. "primary" is the field scored in validation.
CLAUSE_SPECS: dict[str, dict] = {
    "mobilisation_advance": dict(
        label="Mobilisation advance",
        hint="mobilisation advance / advance payment / interest-bearing advance paid to the contractor at the start",
        fields={
            "percent": "advance as % of contract value (number)",
            "interest_rate_pct": "interest charged on the advance, % per annum (0 if interest-free)",
            "recovery_start_pct": "cumulative % of work done when recovery starts",
            "recovery_end_pct": "cumulative % of work done by which it is fully recovered",
        },
        primary="percent",
    ),
    "retention": dict(
        label="Retention / security deposit",
        hint="retention money or security deposit deducted from each running/interim bill",
        fields={
            "percent_per_bill": "% deducted from each gross bill",
            "cap_pct_of_contract": "maximum total retention as % of contract value (null if no cap)",
            "release_at_completion_pct": "share of the retained amount released at completion (0 if all held until end of defect liability)",
        },
        primary="percent_per_bill",
    ),
    "billing_cycle": dict(
        label="Billing cycle",
        hint="how often running / interim payment bills are raised",
        fields={"months": "interval between bills in months (1 = monthly)"},
        primary="months",
    ),
    "payment_period": dict(
        label="Payment period",
        hint="time within which the employer must pay a bill after it is submitted or certified",
        fields={"days": "calendar days; if stated in working days, multiply by 7/5 and say so in note"},
        primary="days",
    ),
    "liquidated_damages": dict(
        label="Liquidated damages",
        hint="compensation / damages for delay in COMPLETION of the works by the contractor. "
             "NOT damages for late handover of site, late performance security, or pre-appointed-date delays",
        fields={
            "rate_pct": "rate as % of contract value",
            "rate_unit": "one of per_day, per_week, per_month",
            "cap_pct": "maximum total damages as % of contract value",
        },
        primary="rate_pct",
    ),
    "price_variation": dict(
        label="Price variation / escalation",
        hint="price variation, price adjustment or escalation clause (e.g. CPWD 10CC, EPC 19.10). "
             "Marked 'deleted' or 'not applicable' means applicable = false",
        fields={
            "applicable": "true or false",
            "coverage_pct": "share of input-cost escalation the contractor recovers (e.g. 85 if 15% is a fixed non-adjustable part); null if not applicable",
        },
        primary="applicable",
    ),
    "performance_bank_guarantee": dict(
        label="Performance bank guarantee",
        hint="performance guarantee / performance security furnished by the contractor",
        fields={"percent": "% of contract value"},
        primary="percent",
    ),
    "defect_liability": dict(
        label="Defect liability period",
        hint="defect liability period / maintenance period after completion",
        fields={"months": "length in months"},
        primary="months",
    ),
}

KEYWORDS = [
    "advance", "mobili", "retention", "security deposit", "performance guarantee", "performance security",
    "bank guarantee", "compensation for delay", "damages", "liquidated", "escalation", "price variation",
    "price adjustment", "10cc", "defect", "maintenance period", "running bill", "interim payment",
    "payment", "schedule f", "stage payment",
]


# ---------------------------------------------------------------- PDF handling
def read_pdf_pages(data: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(data))
    return [(page.extract_text() or "") for page in reader.pages]


def build_context(pages: list[str], max_chars: int = MAX_CHARS) -> tuple[str, list[int]]:
    """Page-tagged text for the prompt. If too long, keep keyword pages and their neighbours."""
    keep = list(range(len(pages)))
    if sum(len(p) for p in pages) > max_chars:
        hits = {i for i, p in enumerate(pages) if any(k in p.lower() for k in KEYWORDS)}
        keep = sorted({j for i in hits for j in (i - 1, i, i + 1) if 0 <= j < len(pages)})
        # still too long: keep the pages with the most keyword hits
        while sum(len(pages[i]) for i in keep) > max_chars and keep:
            score = {i: sum(pages[i].lower().count(k) for k in KEYWORDS) for i in keep}
            keep.remove(min(keep, key=lambda i: score[i]))
    text = "\n".join(f"\n=== PAGE {i + 1} ===\n{pages[i]}" for i in keep)
    return text, [i + 1 for i in keep]


# ---------------------------------------------------------------- prompt + call
def build_prompt(context: str) -> str:
    spec_lines = []
    for key, s in CLAUSE_SPECS.items():
        fields = "; ".join(f'"{f}": {d}' for f, d in s["fields"].items())
        spec_lines.append(f'- "{key}" ({s["hint"]}). values: {{{fields}}}')
    specs = "\n".join(spec_lines)
    return f"""You are reading an Indian construction contract or tender document for a contractor's bid team.
Extract the following financial clauses. The text below is split into pages marked "=== PAGE n ===".

Clauses:
{specs}

Rules:
1. Only report what the document states. If a clause is absent, set "found": false and every value to null. Never guess.
2. "quote" must be copied VERBATIM from the page you cite (max 50 words), so it can be found by text search.
3. "page" is the PAGE number from the markers, not a printed page number.
4. If a general condition is overridden by a project-specific schedule (e.g. Schedule F, SCC, contract data), report the project-specific value and cite that page.
5. Numbers as plain numbers (5 not "5%"). Put any conversion or doubt in "note".

Return only JSON in exactly this shape:
{{"clauses": {{"<clause key>": {{"found": true, "values": {{...}}, "page": 12, "clause_ref": "Clause 10B", "quote": "...", "confidence": "high|medium|low", "note": "..."}}, ...}}}}
Include all {len(CLAUSE_SPECS)} clause keys.

DOCUMENT:
{context}
"""


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The model did not return JSON.")
    return json.loads(text[start:end + 1])


def call_gemini(prompt: str, api_key: str, model: str = DEFAULT_MODEL) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    )
    if not resp.text:
        raise RuntimeError("Gemini returned an empty response (possibly blocked or over quota).")
    return resp.text


def list_models(api_key: str) -> list[str]:
    from google import genai

    client = genai.Client(api_key=api_key)
    return sorted(m.name.replace("models/", "") for m in client.models.list() if "gemini" in m.name)


# ---------------------------------------------------------------- citation check
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%.]+", " ", (s or "").lower())).strip()


def check_quote(quote: str | None, page: int | None, pages: list[str]) -> dict:
    """Is the quoted text really on the cited page? Returns status, score and where it was found."""
    q = _norm(quote)
    if not q:
        return dict(status="no quote", score=0, found_on=None)
    norm_pages = [_norm(p) for p in pages]

    def score_on(i: int) -> float:
        return 100.0 if q in norm_pages[i] else fuzz.partial_ratio(q, norm_pages[i])

    if page and 1 <= page <= len(pages):
        s = score_on(page - 1)
        if s >= 100:
            return dict(status="verbatim on cited page", score=100, found_on=page)
        if s >= 88:
            return dict(status="near-match on cited page", score=round(s), found_on=page)
    best_i, best_s = None, 0.0
    for i in range(len(pages)):
        s = score_on(i)
        if s > best_s:
            best_i, best_s = i, s
    if best_s >= 88:
        return dict(status=f"wrong page (found on p.{best_i + 1})", score=round(best_s), found_on=best_i + 1)
    return dict(status="not found in document", score=round(best_s), found_on=None)


def extract(pdf_bytes: bytes, file_name: str, api_key: str, model: str = DEFAULT_MODEL) -> dict:
    pages = read_pdf_pages(pdf_bytes)
    n_chars = sum(len(p) for p in pages)
    if n_chars < 200 * max(len(pages), 1) * 0.1:
        raise ValueError("Very little text found. This looks like a scanned PDF; run OCR first (known limitation).")
    context, sent_pages = build_context(pages)
    raw = call_gemini(build_prompt(context), api_key, model)
    data = parse_json(raw)
    clauses = data.get("clauses", data)
    for key in CLAUSE_SPECS:
        cl = clauses.setdefault(key, {"found": False, "values": {}, "page": None, "quote": None})
        cl["check"] = check_quote(cl.get("quote"), cl.get("page"), pages) if cl.get("found") else dict(status="not found by AI", score=0, found_on=None)
    return dict(
        source=file_name, model=model, extracted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_pages=len(pages), pages_sent=sent_pages, chars_sent=len(context), clauses=clauses, raw_response=raw,
    )


def recheck(extraction: dict, pdf_bytes: bytes) -> dict:
    """Re-run the citation check (e.g. when a saved extraction is reloaded with its PDF)."""
    pages = read_pdf_pages(pdf_bytes)
    for cl in extraction["clauses"].values():
        if cl.get("found"):
            cl["check"] = check_quote(cl.get("quote"), cl.get("page"), pages)
    return extraction
