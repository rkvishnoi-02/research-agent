"""Service interfaces and default LangChain-friendly implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool

from .models import CompetitorProfile, QueryBank, RawQuote, ResearchQuery, ResearchRequest, VoiceFingerprint
from .quality import filter_query_bank, keyword_summary


def _build_query(text: str, query_type: str, source: str, category: str, intent: str) -> ResearchQuery:
    return ResearchQuery(
        query_text=text,
        query_type=query_type,
        target_source=source,
        expected_category=category,
        intent_target=intent,
    )


def default_query_generator(request: ResearchRequest, loop_count: int, retry_instructions: list[str]) -> QueryBank:
    product = request.product_name
    audience = request.audience
    stronger = loop_count > 0 or any("stronger" in item.lower() for item in retry_instructions)
    modifier = "honest" if stronger else "real"
    pain_word = "annoying" if stronger else "frustrating"
    regret_word = "waste of time" if stronger else "regret"

    bank = QueryBank(
        pain_queries=[
            _build_query(f"{product} {pain_word} for {audience}", "pain", "reddit", "pain", "frustration"),
            _build_query(f"{product} {regret_word} setup", "pain", "web", "pain", "regret"),
            _build_query(f"{product} hate using sequences", "pain", "reddit", "pain", "frustration"),
            _build_query(f"{product} broken workflow {modifier}", "pain", "g2", "pain", "frustration"),
        ],
        complaint_queries=[
            _build_query(f"worst thing about {product} waste of time", "complaint", "reddit", "failed_attempt", "frustration"),
            _build_query(f"{product} not worth it for our team", "complaint", "g2", "objection", "regret"),
            _build_query(f"{product} waste of money after setup", "complaint", "web", "objection", "regret"),
        ],
        switching_queries=[
            _build_query(f"why I left {product}", "switching", "reddit", "switching_story", "switching"),
            _build_query(f"switched from {product} after renewal", "switching", "web", "switching_story", "switching"),
            _build_query(f"moved off {product} honest experience", "switching", "reddit", "switching_story", "switching"),
        ],
        comparison_queries=[
            _build_query(f"{product} vs alternative {modifier}", "comparison", "reddit", "comparison", "comparison"),
            _build_query(f"{product} vs competitor honest review", "comparison", "g2", "comparison", "comparison"),
            _build_query(f"left {product} for competitor comparison", "comparison", "web", "comparison", "comparison"),
        ],
    )
    return filter_query_bank(bank)


def default_voc_collector(request: ResearchRequest, query_bank: QueryBank, loop_count: int) -> list[RawQuote]:
    if request.seed_quotes:
        return request.seed_quotes
    return []


def default_product_understanding(request: ResearchRequest) -> str:
    return f"{request.product_name} targets {request.audience}. Focus: {request.research_request}"


def default_competitor_analysis(request: ResearchRequest) -> list[CompetitorProfile]:
    return [CompetitorProfile(name=name) for name in request.seed_competitors]


def default_voice_analysis(quotes: list[RawQuote]) -> VoiceFingerprint:
    keywords = keyword_summary(quotes)
    buyer_language = [word for word in keywords if word not in {"platform", "solution", "enablement"}][:6]
    fake_language = [word for word in keywords if word in {"platform", "solution", "enablement"}][:4]
    triggers = [word for word in keywords if word in {"waste", "broken", "annoying", "renewal", "setup"}][:4]
    return VoiceFingerprint(
        category_language=keywords[:6],
        buyer_language=buyer_language,
        fake_language=fake_language,
        emotional_triggers=triggers,
        jargon_map={"enablement": "actually usable", "platform": "tool"},
    )


@dataclass
class ExternalResearchAdapters:
    web_search: Callable[[str], str] = lambda query: ""
    forum_search: Callable[[str], str] = lambda query: ""
    fetch_page: Callable[[str], str] = lambda url: ""


def build_langchain_collection_tools(adapters: ExternalResearchAdapters) -> list[StructuredTool]:
    """LangChain tool interfaces for external collection systems."""

    return [
        StructuredTool.from_function(
            name="search_web_research",
            description=(
                "Search the web for emotionally loaded buyer complaints, regret, switching stories, "
                "and honest comparisons. Do not use for neutral product overviews."
            ),
            func=adapters.web_search,
        ),
        StructuredTool.from_function(
            name="search_forum_research",
            description=(
                "Search forums and communities for verbatim buyer language with frustration, regret, "
                "or switching context."
            ),
            func=adapters.forum_search,
        ),
        StructuredTool.from_function(
            name="fetch_source_page",
            description="Fetch source page content to preserve exact quote wording and surrounding context.",
            func=adapters.fetch_page,
        ),
    ]


def create_optional_structured_agent(model: str, system_prompt: str, tools: list[StructuredTool]):
    """Small helper for future LangChain-powered reasoning nodes."""

    return create_agent(model=model, tools=tools, system_prompt=system_prompt)


@dataclass
class ResearchServices:
    query_generator: Callable[[ResearchRequest, int, list[str]], QueryBank] = default_query_generator
    voc_collector: Callable[[ResearchRequest, QueryBank, int], list[RawQuote]] = default_voc_collector
    product_understanding: Callable[[ResearchRequest], str] = default_product_understanding
    competitor_analysis: Callable[[ResearchRequest], list[CompetitorProfile]] = default_competitor_analysis
    voice_analysis: Callable[[list[RawQuote]], VoiceFingerprint] = default_voice_analysis
    collection_adapters: ExternalResearchAdapters = field(default_factory=ExternalResearchAdapters)
