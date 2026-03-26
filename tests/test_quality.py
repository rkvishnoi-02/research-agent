from research_agent.models import QueryBank, RawQuote, ResearchQuery
from research_agent.quality import audit_quote_human_test, filter_query_bank, quote_authenticity_metrics


def test_filter_query_bank_rejects_neutral_queries():
    bank = QueryBank(
        pain_queries=[
            ResearchQuery(
                query_text="features of Apollo",
                query_type="pain",
                target_source="web",
                expected_category="pain",
                intent_target="frustration",
            ),
            ResearchQuery(
                query_text="Apollo annoying setup waste of time",
                query_type="pain",
                target_source="reddit",
                expected_category="pain",
                intent_target="frustration",
            ),
        ]
    )

    filtered = filter_query_bank(bank)

    assert len(filtered.pain_queries) == 1
    assert filtered.pain_queries[0].query_text == "Apollo annoying setup waste of time"
    assert filtered.rejected_queries[0].rejection_reason


def test_human_test_rejects_clean_summary_quotes():
    quote = RawQuote(
        text="Users feel the platform is difficult to use.",
        source="G2",
        source_url="https://example.com/review",
        source_type="g2",
        context="Review summary",
        context_snippet="Reviewer summary\nUsers feel the platform is difficult to use.\nOverall sentiment",
        category="pain",
    )

    audit = audit_quote_human_test(quote)

    assert not audit.passed
    assert any("generic" in reason or "summary" in reason for reason in audit.reasons)


def test_human_test_accepts_specific_buyer_language():
    quote = RawQuote(
        text="We spent three weeks setting this up and half the reps still went back to Gmail because the sequences kept breaking.",
        source="r/sales",
        source_url="https://reddit.com/example",
        source_type="reddit",
        context="Switching discussion",
        context_snippet="I wanted to love it.\nWe spent three weeks setting this up and half the reps still went back to Gmail because the sequences kept breaking.\nThat was the moment we started looking elsewhere.",
        category="pain",
    )

    audit = audit_quote_human_test(quote)

    assert audit.passed
    assert audit.reasons == []


def test_quote_authenticity_metrics_flags_generic_ratio():
    strong = RawQuote(
        text="Our SDR manager killed it after renewal because nobody wanted to babysit the workflows anymore.",
        source="Forum",
        source_url="https://forum.example.com/post",
        source_type="forum",
        context="Renewal complaint",
        context_snippet="The renewal call went badly.\nOur SDR manager killed it after renewal because nobody wanted to babysit the workflows anymore.\nThat was it for us.",
        category="switching_story",
    )
    generic = RawQuote(
        text="Many people say it is useful.",
        source="Web",
        source_url="https://web.example.com/post",
        source_type="web",
        context="Blog summary",
        context_snippet="Summary:\nMany people say it is useful.\nEnd summary.",
        category="pain",
    )

    ratio, generic_ids, failures, _ = quote_authenticity_metrics([strong, generic])

    assert ratio == 0.5
    assert generic.quote_id in generic_ids
    assert failures
