"""Deterministic quality gates for queries and quotes."""

from __future__ import annotations

import re
from collections import Counter

from .models import QueryBank, QuoteAuditResult, RawQuote, ResearchQuery

NEUTRAL_QUERY_PATTERNS = [
    r"\bfeatures? of\b",
    r"\bbenefits? of\b",
    r"\bwhat is\b",
    r"\bpricing\b",
    r"\breview of\b",
]

EMOTIONAL_QUERY_PATTERNS = [
    r"\bannoying\b",
    r"\bfrustrating\b",
    r"\bhate\b",
    r"\bregret\b",
    r"\bwaste of time\b",
    r"\bwaste of money\b",
    r"\bwhy i left\b",
    r"\bwhy we left\b",
    r"\bswitched from\b",
    r"\bvs\b",
    r"\bhonest\b",
    r"\bnot worth it\b",
    r"\bi just want\b",
    r"\bi wish\b",
]

GENERIC_QUOTE_PHRASES = [
    "users feel",
    "customers say",
    "many people",
    "people feel",
    "users say that",
]

LOW_CONTEXT_HINTS = [
    "it helps",
    "it's good",
    "works well",
    "easy to use",
]

EMOTIONAL_WORDS = {
    "annoying",
    "frustrating",
    "hate",
    "awful",
    "waste",
    "regret",
    "stuck",
    "pain",
    "mess",
    "broken",
    "love",
    "best",
    "terrible",
    "impossible",
    "want",
}


def score_query_emotional_intent(query_text: str) -> tuple[float, list[str]]:
    lowered = query_text.lower().strip()
    reasons: list[str] = []
    score = 0.0

    for pattern in EMOTIONAL_QUERY_PATTERNS:
        if re.search(pattern, lowered):
            score += 0.2

    if "reddit" in lowered or "honest" in lowered:
        score += 0.1
    if "left" in lowered or "switched" in lowered:
        score += 0.2
    if "vs" in lowered:
        score += 0.15

    for pattern in NEUTRAL_QUERY_PATTERNS:
        if re.search(pattern, lowered):
            reasons.append("neutral or informational phrasing")
            score -= 0.5

    if score <= 0:
        reasons.append("missing emotional or decision-rich language")

    return max(0.0, min(score, 1.0)), reasons


def enforce_query_rules(query: ResearchQuery) -> ResearchQuery:
    score, reasons = score_query_emotional_intent(query.query_text)
    query.emotional_signal_score = score
    if reasons:
        query.rejection_reason = "; ".join(sorted(set(reasons)))
    return query


def filter_query_bank(query_bank: QueryBank) -> QueryBank:
    kept: dict[str, list[ResearchQuery]] = {
        "pain_queries": [],
        "complaint_queries": [],
        "switching_queries": [],
        "comparison_queries": [],
        "desire_queries": [],
    }

    rejected = list(query_bank.rejected_queries)
    seen: set[str] = set(query_bank.executed_queries)

    for field in kept:
        for query in getattr(query_bank, field):
            query = enforce_query_rules(query)
            normalized = query.query_text.lower().strip()
            if normalized in seen:
                query.rejection_reason = "duplicate query"
                rejected.append(query)
                continue
            seen.add(normalized)
            if query.rejection_reason:
                rejected.append(query)
                continue
            kept[field].append(query)

    return QueryBank(
        **kept,
        executed_queries=query_bank.executed_queries,
        rejected_queries=rejected,
    )


def audit_quote_human_test(quote: RawQuote) -> QuoteAuditResult:
    reasons: list[str] = []
    lowered = quote.text.lower().strip()
    context_lowered = quote.context_snippet.lower().strip()

    if any(phrase in lowered for phrase in GENERIC_QUOTE_PHRASES):
        reasons.append("contains generic proxy phrasing")

    if len(lowered.split()) < 5:
        reasons.append("too short to be a concrete conversational quote")

    if any(hint in lowered for hint in LOW_CONTEXT_HINTS) and not any(
        token in lowered for token in ("because", "when", "after", "during", "spent", "month", "week", "renewal")
    ):
        reasons.append("lacks a concrete situation")

    if quote.text.endswith(":") or quote.text.lower().startswith(("users", "customers", "people")):
        reasons.append("sounds like a cleaned-up summary")

    if not context_lowered:
        reasons.append("missing context snippet")

    if quote.source_type == "unknown":
        reasons.append("missing source type")

    emotional_hits = sum(word in lowered for word in EMOTIONAL_WORDS)
    specificity_hits = sum(
        token in lowered
        for token in ("because", "when", "after", "during", "renewal", "setup", "team", "month", "week", "$", "ticket", "breaks", "reps")
    )
    generic = bool(reasons)
    if emotional_hits == 0 and specificity_hits == 0:
        reasons.append("lacks emotional tone and specific context")
        generic = True

    return QuoteAuditResult(
        quote_id=quote.quote_id,
        passed=not reasons,
        reasons=reasons,
        generic=generic or bool(reasons),
    )


def quote_authenticity_metrics(quotes: list[RawQuote]) -> tuple[float, list[str], list[str], list[QuoteAuditResult]]:
    if not quotes:
        return 1.0, [], ["no quotes collected"], []

    audits = [audit_quote_human_test(quote) for quote in quotes]
    generic_ids = [audit.quote_id for audit in audits if audit.generic]
    failures = [f"{audit.quote_id}: {', '.join(audit.reasons)}" for audit in audits if audit.reasons]
    ratio = len(generic_ids) / len(quotes)
    return ratio, generic_ids, failures, audits


def keyword_summary(quotes: list[RawQuote], top_k: int = 10) -> list[str]:
    counter: Counter[str] = Counter()
    for quote in quotes:
        for token in re.findall(r"[a-zA-Z]{4,}", quote.text.lower()):
            if token not in {"that", "with", "this", "have", "your", "just", "they", "them"}:
                counter[token] += 1
    return [word for word, _ in counter.most_common(top_k)]
