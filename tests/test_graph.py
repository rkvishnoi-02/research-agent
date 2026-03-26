from langgraph.checkpoint.memory import InMemorySaver

from research_agent.graph import build_research_graph
from research_agent.models import QueryBank, RawQuote, ResearchQuery, ResearchRequest, ValidationThresholds
from research_agent import services as service_module
from research_agent.services import ResearchServices


class StatefulCollector:
    def __init__(self):
        self.calls = 0

    def __call__(self, request, query_bank, loop_count):
        self.calls += 1
        if self.calls == 1:
            return [
                RawQuote(
                    text="Many people say this tool is useful.",
                    source="Web",
                    source_url="https://example.com/summary",
                    source_type="web",
                    context="Roundup article",
                    context_snippet="Roundup intro\nMany people say this tool is useful.\nRoundup outro",
                    category="pain",
                ),
                RawQuote(
                    text="We burned a month trying to get sequences live and still had reps sending manual follow-ups from Gmail.",
                    source="r/sales",
                    source_url="https://reddit.com/post1",
                    source_type="reddit",
                    context="Setup frustration",
                    context_snippet="The rollout was brutal.\nWe burned a month trying to get sequences live and still had reps sending manual follow-ups from Gmail.\nThat was a huge red flag.",
                    category="pain",
                ),
                RawQuote(
                    text="We left after renewal jumped 40% and leadership refused to keep paying for a tool the team avoided.",
                    source="G2",
                    source_url="https://g2.com/review1",
                    source_type="g2",
                    context="Switching story",
                    context_snippet="The renewal was the breaking point.\nWe left after renewal jumped 40% and leadership refused to keep paying for a tool the team avoided.\nWe moved the team the next month.",
                    category="switching_story",
                ),
            ]

        return [
            RawQuote(
                text="I regret buying this because our ops lead spent every Friday fixing broken routing rules instead of coaching reps.",
                source="Forum",
                source_url="https://forum.example.com/post2",
                source_type="forum",
                context="Regret thread",
                context_snippet="I regret the purchase.\nI regret buying this because our ops lead spent every Friday fixing broken routing rules instead of coaching reps.\nThat cost us real pipeline time.",
                category="objection",
            ),
            RawQuote(
                text="We switched to a simpler tool after two quarters because nobody on the team trusted the reporting anymore.",
                source="r/revops",
                source_url="https://reddit.com/post2",
                source_type="reddit",
                context="Switching story",
                context_snippet="Trust disappeared.\nWe switched to a simpler tool after two quarters because nobody on the team trusted the reporting anymore.\nThat pushed the VP to approve the move.",
                category="switching_story",
            ),
            RawQuote(
                text="Once we ripped out the fancy AI copy and went back to plain templates, reply quality actually improved.",
                source="G2",
                source_url="https://g2.com/review2",
                source_type="g2",
                context="Comparison feedback",
                context_snippet="We tested both approaches.\nOnce we ripped out the fancy AI copy and went back to plain templates, reply quality actually improved.\nThat made the premium add-on feel pointless.",
                category="comparison",
            ),
            RawQuote(
                text="I just want a follow-up tool that my reps can use without filing a ticket every time a sequence breaks.",
                source="Hacker News",
                source_url="https://news.ycombinator.com/item?id=1",
                source_type="hackernews",
                context="Desire statement",
                context_snippet="This is all I want.\nI just want a follow-up tool that my reps can use without filing a ticket every time a sequence breaks.\nWhy is that impossible?",
                category="desire",
            ),
            RawQuote(
                text="This was the best reporting we had after setup, but the setup itself was a nightmare and nobody wanted to own it.",
                source="G2",
                source_url="https://g2.com/review3",
                source_type="g2",
                context="Mixed experience",
                context_snippet="There was one upside.\nThis was the best reporting we had after setup, but the setup itself was a nightmare and nobody wanted to own it.\nThat tension never really went away.",
                category="comparison",
            ),
        ]


def test_graph_retries_after_generic_quote_failure_and_recovers():
    collector = StatefulCollector()

    def fixed_query_generator(request, loop_count, retry_instructions):
        return QueryBank(
            pain_queries=[
                ResearchQuery(
                    query_text=f"tool annoying waste of time {i}",
                    query_type="pain",
                    target_source="reddit",
                    expected_category="pain",
                    intent_target="frustration",
                )
                for i in range(4)
            ],
            complaint_queries=[
                ResearchQuery(
                    query_text=f"tool regret purchase {i}",
                    query_type="complaint",
                    target_source="g2",
                    expected_category="objection",
                    intent_target="regret",
                )
                for i in range(3)
            ],
            switching_queries=[
                ResearchQuery(
                    query_text=f"why I left tool {i}",
                    query_type="switching",
                    target_source="reddit",
                    expected_category="switching_story",
                    intent_target="switching",
                )
                for i in range(3)
            ],
            comparison_queries=[
                ResearchQuery(
                    query_text=f"tool vs competitor honest {i}",
                    query_type="comparison",
                    target_source="reddit",
                    expected_category="comparison",
                    intent_target="comparison",
                )
                for i in range(3)
            ],
            desire_queries=[
                ResearchQuery(
                    query_text=f"i just want a tool that works {i}",
                    query_type="pain",
                    target_source="reddit",
                    expected_category="desire",
                    intent_target="frustration",
                )
                for i in range(2)
            ],
        )

    services = ResearchServices(
        query_generator=fixed_query_generator,
        voc_collector=collector,
    )
    thresholds = ValidationThresholds(
        min_pain_quotes=1,
        min_failed_attempts=0,
        min_objections=1,
        min_switching_stories=1,
        min_comparisons=1,
        min_desires=0,
        min_contradictions=1,
        min_decision_moments=1,
        min_truth_density=1.0,
        max_generic_quote_ratio=0.30,
        max_loops=2,
    )
    graph = build_research_graph(services=services, thresholds=thresholds, checkpointer=InMemorySaver())
    request = ResearchRequest(
        research_request="Research raw buyer frustration and switching language for a sales engagement platform.",
        product_name="Synvo",
        audience="SDR teams at mid-market SaaS companies",
        industry="B2B SaaS",
        company_size="100-500 employees",
    )

    result = graph.invoke_request(request, config={"configurable": {"thread_id": "graph-retry-test"}})

    assert result["final_research_packet"] is not None
    assert result["loop_count"] == 1
    assert collector.calls == 2
    assert result["generic_quote_ratio"] <= 0.30


def test_graph_rejects_vague_request():
    graph = build_research_graph(checkpointer=InMemorySaver())
    request = ResearchRequest(
        research_request="Research everything",
        product_name="Synvo",
        audience="",
    )

    result = graph.invoke_request(request, config={"configurable": {"thread_id": "graph-vague-test"}})

    assert result["status"] == "rejected"
    assert result["validation_failures"]


def test_query_quality_gate_stops_retrying_and_escalates():
    def weak_query_generator(request, loop_count, retry_instructions):
        return QueryBank(
            pain_queries=[
                ResearchQuery(
                    query_text="tool annoying setup",
                    query_type="pain",
                    target_source="reddit",
                    expected_category="pain",
                    intent_target="frustration",
                )
            ],
            complaint_queries=[],
            switching_queries=[],
            comparison_queries=[],
        )

    services = ResearchServices(query_generator=weak_query_generator)
    thresholds = ValidationThresholds(max_loops=2)
    graph = build_research_graph(services=services, thresholds=thresholds, checkpointer=InMemorySaver())
    request = ResearchRequest(
        research_request="Research raw buyer frustration for a sales engagement platform.",
        product_name="Synvo",
        audience="SDR teams",
    )

    result = graph.invoke_request(request, config={"configurable": {"thread_id": "query-gate-escalate"}})

    assert result["status"] == "escalated"
    assert result["loop_count"] == 2
    assert result["escalation_report"] is not None
    assert any(f.code == "weak_pain_queries" or f.code == "weak_complaint_queries" for f in result["validation_failures"])


def test_voc_collector_falls_back_to_reddit_when_apify_returns_nothing(monkeypatch):
    request = ResearchRequest(
        research_request="Research raw buyer frustration for a sales engagement platform.",
        product_name="Synvo",
        audience="SDR teams",
    )
    bank = QueryBank(
        pain_queries=[
            ResearchQuery(
                query_text="tool annoying reddit",
                query_type="pain",
                target_source="reddit",
                expected_category="pain",
                intent_target="frustration",
            )
        ]
    )
    fallback_quote = RawQuote(
        text="We wasted two weeks trying to get Outreach live and ended up reverting to manual follow-ups.",
        source="r/sales",
        source_url="https://reddit.com/post",
        source_type="reddit",
        context="Launch failure",
        context_snippet="The rollout fell apart.\nWe wasted two weeks trying to get Outreach live and ended up reverting to manual follow-ups.\nThat was the end of the test.",
        category="failed_attempt",
    )

    monkeypatch.setattr(service_module, "_collect_apify_reddit_quotes_batch", lambda queries: [])
    monkeypatch.setattr(service_module, "_collect_reddit_quotes", lambda query, category: [fallback_quote])

    quotes = service_module.default_voc_collector(request, bank, 0)

    assert quotes == [fallback_quote]
