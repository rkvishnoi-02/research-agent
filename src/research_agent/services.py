"""Service interfaces and default LangChain-friendly implementations."""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from typing import Callable

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool

from .models import CompetitorProfile, QueryBank, RawQuote, ResearchQuery, ResearchRequest, VoiceFingerprint
from .quality import filter_query_bank, keyword_summary

USER_AGENT = "Mozilla/5.0 (compatible; ResearchAgent/0.1; +https://github.com/rkvishnoi-02/research-agent)"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"
SOURCE_TYPE_MAP = {
    "reddit.com": "reddit",
    "old.reddit.com": "reddit",
    "news.ycombinator.com": "hackernews",
    "g2.com": "g2",
    "capterra.com": "capterra",
    "trustradius.com": "trust_radius",
}
CONTEXT_AND_EMOTION_HINTS = (
    "because",
    "when",
    "after",
    "during",
    "frustrat",
    "annoy",
    "hate",
    "waste",
    "regret",
    "switched",
    "left",
    "renewal",
    "setup",
    "ticket",
    "broke",
    "broken",
)


def _derive_market_topic(request: ResearchRequest) -> str:
    text = request.research_request.strip()
    lowered = text.lower()
    match = re.search(r"\bfor (?:an?|the)\s+(.+?)(?: targeting| aimed at| for |$)", lowered)
    if match:
        topic = match.group(1).strip(" .")
        return re.sub(r"\s+", " ", topic)
    if "sales engagement" in lowered:
        return "sales engagement platform"
    if request.industry:
        return request.industry
    return request.product_name


def _derive_search_subject(request: ResearchRequest) -> str:
    if request.product_url:
        return request.product_name
    return _derive_market_topic(request)


def _build_query(text: str, query_type: str, source: str, category: str, intent: str) -> ResearchQuery:
    return ResearchQuery(
        query_text=text,
        query_type=query_type,
        target_source=source,
        expected_category=category,
        intent_target=intent,
    )


def default_query_generator(request: ResearchRequest, loop_count: int, retry_instructions: list[str]) -> QueryBank:
    subject = _derive_search_subject(request)
    audience = request.audience
    competitors = _default_competitor_seeds(subject)
    primary = competitors[0] if competitors else subject
    secondary = competitors[1] if len(competitors) > 1 else "alternative"
    tertiary = competitors[2] if len(competitors) > 2 else subject
    comparison_subject = " vs ".join(competitors) if competitors else f"{subject} vs alternative"
    switching_subject = primary
    stronger = loop_count > 0 or any("stronger" in item.lower() for item in retry_instructions)
    modifier = "honest" if stronger else "real"
    pain_word = "annoying" if stronger else "frustrating"
    regret_word = "waste of time" if stronger else "regret"

    bank = QueryBank(
        pain_queries=[
            _build_query(f"{primary} {pain_word} for {audience} reddit", "pain", "reddit", "pain", "frustration"),
            _build_query(f"{secondary} {regret_word} setup reddit", "pain", "reddit", "pain", "regret"),
            _build_query(f"{tertiary} hate using sequences reddit", "pain", "reddit", "pain", "frustration"),
            _build_query(f"{primary} broken workflow honest review", "pain", "web", "pain", "frustration"),
        ],
        complaint_queries=[
            _build_query(f"worst thing about {primary} waste of time reddit", "complaint", "reddit", "failed_attempt", "frustration"),
            _build_query(f"{secondary} not worth it for our team reddit", "complaint", "reddit", "objection", "regret"),
            _build_query(f"{tertiary} waste of money after setup reddit", "complaint", "reddit", "objection", "regret"),
        ],
        switching_queries=[
            _build_query(f"why I left {switching_subject} reddit", "switching", "reddit", "switching_story", "switching"),
            _build_query(f"switched from {secondary} after renewal reddit", "switching", "reddit", "switching_story", "switching"),
            _build_query(f"moved off {tertiary} honest experience reddit", "switching", "reddit", "switching_story", "switching"),
        ],
        comparison_queries=[
            _build_query(f"{comparison_subject} honest reddit", "comparison", "reddit", "comparison", "comparison"),
            _build_query(f"{comparison_subject} honest comparison reddit", "comparison", "reddit", "comparison", "comparison"),
            _build_query(f"{comparison_subject} for small team reddit", "comparison", "reddit", "comparison", "comparison"),
        ],
        desire_queries=[
            _build_query(f'i just want a {subject} that works reddit', "pain", "reddit", "desire", "frustration"),
            _build_query(f'i wish {subject} did not need admin setup reddit', "pain", "reddit", "desire", "frustration"),
        ],
    )
    return filter_query_bank(bank)


def default_voc_collector(request: ResearchRequest, query_bank: QueryBank, loop_count: int) -> list[RawQuote]:
    if request.seed_quotes:
        return request.seed_quotes
    quotes: list[RawQuote] = []
    seen_texts: set[str] = set()
    max_queries = len(query_bank.all_queries)
    for query in query_bank.all_queries[:max_queries]:
        source_candidates: list[RawQuote] = []
        if query.target_source == "reddit":
            source_candidates = _collect_reddit_quotes(query.query_text, query.expected_category)
        elif query.target_source == "hn":
            source_candidates = _collect_hn_quotes(query.query_text, query.expected_category)
        else:
            source_candidates = _collect_web_quotes(query.query_text, query.expected_category)

        for candidate in source_candidates:
            norm = candidate.text.strip().lower()
            if norm in seen_texts:
                continue
            seen_texts.add(norm)
            quotes.append(candidate)
            if len(quotes) >= 32:
                return quotes
    return quotes


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


def _normalize_result_url(url: str) -> str:
    if "duckduckgo.com/l/?" in url:
        parsed = urlparse(url)
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return url


def _source_type_for_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    for domain, source_type in SOURCE_TYPE_MAP.items():
        if host.endswith(domain):
            return source_type
    return "web"


def _default_competitor_seeds(subject: str) -> list[str]:
    lowered = subject.lower()
    if "sales engagement" in lowered:
        return ["Outreach", "Salesloft", "Apollo"]
    if "crm" in lowered:
        return ["HubSpot", "Salesforce", "Pipedrive"]
    return []


@lru_cache(maxsize=128)
def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict[str, str]]:
    tavily_results = _search_tavily(query, max_results=max_results)
    if tavily_results:
        return tavily_results
    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    results: list[dict[str, str]] = []
    for anchor in soup.select("a.result__a"):
        href = anchor.get("href") or ""
        url = _normalize_result_url(href)
        if not url.startswith("http"):
            continue
        results.append({"title": anchor.get_text(" ", strip=True), "url": url})
        if len(results) >= max_results:
            break
    return results


def _search_tavily(query: str, max_results: int = 5) -> list[dict[str, str]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "topic": "general",
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return []
    results = []
    for item in payload.get("results", []):
        url = item.get("url")
        if not url:
            continue
        results.append({"title": item.get("title", urlparse(url).netloc), "url": url})
    return results


@lru_cache(maxsize=128)
def _fetch_page_lines(url: str) -> tuple[str, ...]:
    firecrawl_lines = _fetch_with_firecrawl(url)
    if firecrawl_lines:
        return tuple(firecrawl_lines)
    readers = [
        f"https://r.jina.ai/http://{url}",
        f"https://r.jina.ai/http://{url.replace('https://', '').replace('http://', '')}",
        url,
    ]
    for candidate in readers:
        try:
            response = requests.get(candidate, headers={"User-Agent": USER_AGENT}, timeout=25)
            if response.ok and response.text:
                lines = _html_to_lines(response.text)
                if lines:
                    return tuple(lines)
        except requests.RequestException:
            continue
    return tuple()


def _fetch_with_firecrawl(url: str) -> list[str]:
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return []
    try:
        response = requests.post(
            FIRECRAWL_SCRAPE_URL,
            json={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            timeout=40,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return []

    data = payload.get("data", {})
    markdown = data.get("markdown", "")
    if not markdown:
        return []
    return _html_to_lines(markdown)


def _html_to_lines(raw_text: str) -> list[str]:
    text = raw_text
    if "<html" in raw_text.lower():
        soup = BeautifulSoup(raw_text, "html.parser")
        text = soup.get_text("\n", strip=True)
    text = html.unescape(text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return [line for line in lines if len(line.split()) >= 6]


def _looks_like_real_quote(line: str) -> bool:
    lowered = line.lower()
    if len(line) > 320:
        return False
    if line.count("http") or lowered.startswith(("sign in", "cookie", "accept", "menu", "share", "* ", "- ", "title:", "author:")):
        return False
    if any(phrase in lowered for phrase in ("users feel", "customers say", "many people")):
        return False
    if not any(token in lowered for token in (" i ", " we ", " my ", " our ", " me ", " us ", " i'm", " i've", " we'd", " we're")):
        return False
    if any(token in lowered for token in CONTEXT_AND_EMOTION_HINTS):
        return True
    if " i " in f" {lowered} " or " we " in f" {lowered} ":
        return True
    return False


def _infer_quote_category(line: str, fallback: str) -> str:
    lowered = line.lower()
    if any(token in lowered for token in ("i just want", "i wish", "wanted a", "wish there was")):
        return "desire"
    if any(token in lowered for token in ("why i left", "we left", "i switched", "we switched", "moved off", "moved to")):
        return "switching_story"
    if any(token in lowered for token in (" vs ", "compared to", "alternative", "better than", "salesloft", "outreach", "apollo")):
        return "comparison"
    if any(token in lowered for token in ("not worth it", "too expensive", "price", "pricing", "renewal")):
        return "objection"
    if any(token in lowered for token in ("we tried", "i tried", "didn't work", "did not work", "failed", "gave up")):
        return "failed_attempt"
    if any(token in lowered for token in ("love", "best thing", "worked great")):
        return "delight"
    return fallback


def _extract_quote_candidates(lines: tuple[str, ...], category: str, url: str, title: str) -> list[RawQuote]:
    candidates: list[RawQuote] = []
    source_type = _source_type_for_url(url)
    for index, line in enumerate(lines):
        if not _looks_like_real_quote(line):
            continue
        start = max(0, index - 1)
        end = min(len(lines), index + 2)
        context_snippet = "\n".join(lines[start:end])
        candidates.append(
            RawQuote(
                text=line,
                source=title or urlparse(url).netloc,
                source_url=url,
                source_type=source_type,
                context=title or urlparse(url).netloc,
                context_snippet=context_snippet,
                category=_infer_quote_category(line, category),
            )
        )
        if len(candidates) >= 3:
            break
    return candidates


def _collect_web_quotes(query: str, category: str) -> list[RawQuote]:
    candidates: list[RawQuote] = []
    for result in _search_duckduckgo(query, max_results=3):
        page_lines = _fetch_page_lines(result["url"])
        if not page_lines:
            continue
        candidates.extend(_extract_quote_candidates(page_lines, category, result["url"], result["title"]))
        if len(candidates) >= 6:
            break
    return candidates


def _collect_reddit_quotes(query: str, category: str) -> list[RawQuote]:
    candidates: list[RawQuote] = []
    for result in _search_duckduckgo(f"site:reddit.com {query}", max_results=4):
        parsed = urlparse(result["url"])
        if "reddit.com" not in parsed.netloc:
            continue
        path = parsed.path.rstrip("/")
        if "/comments/" not in path:
            continue
        json_url = f"https://www.reddit.com{path}.json"
        try:
            comments_response = requests.get(
                json_url,
                params={"limit": 10, "sort": "top", "raw_json": 1},
                headers={"User-Agent": USER_AGENT},
                timeout=20,
            )
            comments_response.raise_for_status()
            comments_payload = comments_response.json()
        except requests.RequestException:
            continue

        title = result["title"]
        text_blocks: list[str] = []
        if comments_payload:
            post_listing = comments_payload[0].get("data", {}).get("children", [])
            if post_listing:
                post_data = post_listing[0].get("data", {})
                if post_data.get("selftext"):
                    text_blocks.append(post_data["selftext"])
                title = post_data.get("title") or title
            if len(comments_payload) > 1:
                for comment in comments_payload[1].get("data", {}).get("children", []):
                    comment_data = comment.get("data", {})
                    body = comment_data.get("body", "")
                    if body:
                        text_blocks.append(body)

        for block in text_blocks:
            lines = tuple(_html_to_lines(block))
            if not lines:
                continue
            candidates.extend(_extract_quote_candidates(lines, category, result["url"], title))
            if len(candidates) >= 12:
                return candidates
    return candidates


def _collect_hn_quotes(query: str, category: str) -> list[RawQuote]:
    candidates: list[RawQuote] = []
    try:
        response = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "tags": "comment", "hitsPerPage": 8},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return candidates

    for hit in payload.get("hits", []):
        comment_text = hit.get("comment_text") or ""
        if not comment_text:
            continue
        item_id = hit.get("objectID")
        url = f"https://news.ycombinator.com/item?id={item_id}" if item_id else "https://news.ycombinator.com"
        title = hit.get("story_title") or "Hacker News"
        lines = tuple(_html_to_lines(comment_text))
        candidates.extend(_extract_quote_candidates(lines, category, url, title))
        if len(candidates) >= 8:
            break
    return candidates
