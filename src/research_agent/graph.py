"""LangGraph workflow for the research agent."""

from __future__ import annotations

from collections import Counter
from typing import cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from .models import (
    ApprovedResearchPacket,
    Contradiction,
    DecisionMoment,
    EscalationReport,
    EvidenceLink,
    EvidenceMap,
    FailedAttempt,
    Insight,
    QueryBank,
    QuoteAuditResult,
    ReviewCheckpoint,
    StageOneExtraction,
    SynthesisArtifacts,
    ValidationFailure,
    ValidationThresholds,
    VoiceFingerprint,
    ResearchState,
)
from .quality import filter_query_bank, quote_authenticity_metrics
from .retrieval import LexicalEvidenceRetriever
from .services import ResearchServices


def _initial_state() -> dict:
    return {
        "status": "pending",
        "loop_count": 0,
        "retry_instructions": [],
        "candidate_quotes": [],
        "raw_quotes": [],
        "rejected_quotes": [],
        "quote_audits": [],
        "competitors": [],
        "collection_logs": [],
        "validation_failures": [],
        "generic_quote_ids": [],
        "quote_authenticity_failures": [],
    }


def build_research_graph(
    services: ResearchServices | None = None,
    thresholds: ValidationThresholds | None = None,
    checkpointer=None,
):
    services = services or ResearchServices()
    thresholds = thresholds or ValidationThresholds()

    def active_thresholds(state: ResearchState) -> ValidationThresholds:
        request = state["request"]
        if request.research_mode == "strict":
            return thresholds
        return thresholds.model_copy(
            update={
                "min_pain_quotes": min(thresholds.min_pain_quotes, 1),
                "min_failed_attempts": 0,
                "min_objections": 0,
                "min_switching_stories": 0,
                "min_comparisons": 1,
                "min_desires": 0,
                "min_contradictions": 0,
                "min_decision_moments": 0,
                "min_truth_density": min(thresholds.min_truth_density, 1.0),
                "max_loops": min(thresholds.max_loops, 2),
            }
        )

    def strategist_node(state: ResearchState) -> dict:
        request = state["request"]
        failures: list[ValidationFailure] = []
        request_text = request.research_request.lower()
        if len(request.research_request.split()) < 5 or "research everything" in request_text:
            failures.append(
                ValidationFailure(
                    code="vague_input",
                    message="Research request is too vague. Focus the request on a specific audience and angle.",
                    severity="critical",
                )
            )
        if not request.audience.strip():
            failures.append(
                ValidationFailure(
                    code="missing_audience",
                    message="Audience is required for the strategist node.",
                    severity="critical",
                )
            )
        return {
            "status": "rejected" if failures else "running",
            "validation_failures": failures,
        }

    def query_generation_node(state: ResearchState) -> dict:
        request = state["request"]
        query_bank = services.query_generator(
            request,
            state.get("loop_count", 0),
            state.get("retry_instructions", []),
        )
        return {
            "query_bank": filter_query_bank(query_bank),
            "collection_logs": [f"Generated {len(query_bank.all_queries)} candidate queries"],
        }

    def query_quality_gate_node(state: ResearchState) -> dict:
        current_thresholds = active_thresholds(state)
        bank = state["query_bank"] or QueryBank()
        failures: list[ValidationFailure] = []
        if len(bank.pain_queries) < 4:
            failures.append(
                ValidationFailure(
                    code="weak_pain_queries",
                    message="Pain queries failed the emotional/decision-rich requirement.",
                    severity="critical",
                    retry_instruction="Generate stronger frustration and regret queries. Avoid neutral language.",
                )
            )
        if len(bank.complaint_queries) < 3:
            failures.append(
                ValidationFailure(
                    code="weak_complaint_queries",
                    message="Complaint queries are too neutral or informational.",
                    severity="critical",
                    retry_instruction="Generate stronger grievance-focused queries using waste, regret, and annoyance framing.",
                )
            )
        if len(bank.switching_queries) < 3:
            failures.append(
                ValidationFailure(
                    code="weak_switching_queries",
                    message="Switching queries are missing migration intent.",
                    severity="critical",
                    retry_instruction="Generate more why-I-left and switched-from queries.",
                )
            )
        if len(bank.comparison_queries) < 3:
            failures.append(
                ValidationFailure(
                    code="weak_comparison_queries",
                    message="Comparison queries need honest, head-to-head framing.",
                    severity="critical",
                    retry_instruction="Generate stronger honest comparison queries.",
                )
            )
        if len(bank.desire_queries) < 2:
            failures.append(
                ValidationFailure(
                    code="weak_desire_queries",
                    message="Desire queries need explicit wish or I-just-want phrasing.",
                    severity="critical",
                    retry_instruction="Generate desire queries using I wish, I just want, and what I wanted language.",
                )
            )
        instructions = [failure.retry_instruction for failure in failures if failure.retry_instruction]
        next_loop_count = state.get("loop_count", 0) + (1 if failures else 0)
        status = "running"
        if failures:
            status = "retry" if next_loop_count < current_thresholds.max_loops else "escalated"
        return {
            "validation_failures": failures,
            "retry_instructions": instructions,
            "status": status,
            "loop_count": next_loop_count,
        }

    def spawn_collection(state: ResearchState):
        return [
            Send("collect_voc", state),
            Send("collect_product", state),
            Send("collect_competitors", state),
        ]

    def collect_voc_node(state: ResearchState) -> dict:
        request = state["request"]
        query_bank = cast(QueryBank, state["query_bank"])
        quotes = services.voc_collector(request, query_bank, state.get("loop_count", 0))
        return {
            "candidate_quotes": quotes,
            "query_bank": QueryBank(
                pain_queries=query_bank.pain_queries,
                complaint_queries=query_bank.complaint_queries,
                switching_queries=query_bank.switching_queries,
                comparison_queries=query_bank.comparison_queries,
                desire_queries=query_bank.desire_queries,
                executed_queries=[query.query_text for query in query_bank.all_queries],
                rejected_queries=query_bank.rejected_queries,
            ),
            "collection_logs": [f"Collected {len(quotes)} candidate VoC quotes"],
        }

    def collect_product_node(state: ResearchState) -> dict:
        return {
            "product_summary": services.product_understanding(state["request"]),
            "collection_logs": ["Completed product understanding"],
        }

    def collect_competitors_node(state: ResearchState) -> dict:
        competitors = services.competitor_analysis(state["request"])
        return {
            "competitors": competitors,
            "collection_logs": [f"Collected {len(competitors)} competitor profiles"],
        }

    def quote_authenticity_gate_node(state: ResearchState) -> dict:
        seen_ids = {quote.quote_id for quote in state.get("raw_quotes", [])} | {
            quote.quote_id for quote in state.get("rejected_quotes", [])
        }
        seen_texts = {
            quote.text.strip().lower()
            for quote in state.get("raw_quotes", []) + state.get("rejected_quotes", [])
            if quote.text.strip()
        }
        new_candidates = []
        for quote in state.get("candidate_quotes", []):
            normalized = quote.text.strip().lower()
            if quote.quote_id in seen_ids or normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            new_candidates.append(quote)

        accepted = []
        rejected = []
        audits: list[QuoteAuditResult] = []
        ratio = 0.0
        for quote in new_candidates:
            ratio, _, _, audit_items = quote_authenticity_metrics([quote])
            audit = audit_items[0]
            audits.append(audit)
            if audit.passed:
                quote.human_test_passed = True
                quote.human_test_failure_reasons = []
                accepted.append(quote)
            else:
                quote.human_test_passed = False
                quote.human_test_failure_reasons = audit.reasons
                rejected.append(quote)
        return {
            "raw_quotes": accepted,
            "rejected_quotes": rejected,
            "quote_audits": audits,
            "collection_logs": [
                f"Quote authenticity gate accepted {len(accepted)} and rejected {len(rejected)} quotes (seed ratio {ratio:.2f})"
            ],
        }

    def hitl_post_voc_node(state: ResearchState) -> dict:
        request = state["request"]
        if request.hitl_mode == "disabled":
            return {"hitl_post_voc": ReviewCheckpoint(status="auto_approved")}

        response = interrupt(
            {
                "stage": "post_voc_review",
                "quote_count": len(state.get("raw_quotes", [])),
                "rejected_quote_count": len(state.get("rejected_quotes", [])),
                "sample_quote_ids": [quote.quote_id for quote in state.get("raw_quotes", [])[:5]],
            }
        )
        approved = bool(response.get("approved", False))
        return {
            "hitl_post_voc": ReviewCheckpoint(
                status="approved" if approved else "rejected",
                notes=response.get("notes"),
            )
        }

    def voice_fingerprint_node(state: ResearchState) -> dict:
        return {"voice_fingerprint": services.voice_analysis(state.get("raw_quotes", []))}

    def synthesis_stage_one_node(state: ResearchState) -> dict:
        extraction = StageOneExtraction()
        for quote in state.get("raw_quotes", []):
            if quote.category == "pain":
                extraction.extracted_pains.append(quote.quote_id)
            elif quote.category == "desire":
                extraction.extracted_desires.append(quote.quote_id)
            elif quote.category == "objection":
                extraction.extracted_objections.append(quote.quote_id)
            elif quote.category == "switching_story":
                extraction.extracted_switching_stories.append(quote.quote_id)
            elif quote.category == "comparison":
                extraction.extracted_comparisons.append(quote.quote_id)
            elif quote.category == "failed_attempt":
                extraction.extracted_failed_attempts.append(quote.quote_id)
            else:
                extraction.unclassified_quote_ids.append(quote.quote_id)
        return {"stage_one": extraction}

    def synthesis_stage_two_node(state: ResearchState) -> dict:
        quotes = state.get("raw_quotes", [])
        retriever = LexicalEvidenceRetriever.from_quotes(quotes, k=3)
        synthesis = SynthesisArtifacts()

        for quote in quotes:
            if quote.category == "objection":
                synthesis.objections.append(
                    Insight(insight_id=f"obj-{quote.quote_id}", text=quote.text, supporting_quote_ids=[quote.quote_id])
                )
            if quote.category == "failed_attempt":
                synthesis.failed_attempts.append(
                    FailedAttempt(
                        what_they_tried=quote.context or "Tried the product",
                        why_it_failed=quote.text,
                        supporting_quote_ids=[quote.quote_id],
                    )
                )
            if quote.category == "switching_story":
                synthesis.decision_moments.append(
                    DecisionMoment(
                        trigger=quote.context or quote.context_snippet,
                        decision=quote.text,
                        supporting_quote_ids=[quote.quote_id],
                    )
                )

        negative_quotes = [quote for quote in quotes if any(word in quote.text.lower() for word in ("annoying", "hate", "waste", "broken", "terrible"))]
        positive_quotes = [
            quote
            for quote in quotes
            if quote.category == "delight" or any(word in quote.text.lower() for word in ("love", "best", "worth", "great"))
        ]
        if negative_quotes and positive_quotes:
            a = negative_quotes[0]
            b = positive_quotes[0]
            synthesis.contradictions.append(
                Contradiction(
                    quote_id_a=a.quote_id,
                    quote_id_b=b.quote_id,
                    statement_a_text=a.text,
                    statement_b_text=b.text,
                    tension="The market shows a split between frustration and positive payoff after setup.",
                    signal_strength="moderate",
                )
            )

        links: list[EvidenceLink] = []
        for insight in synthesis.objections:
            docs = retriever.invoke(insight.text)
            supporting_ids = [document.metadata["quote_id"] for document in docs if document.metadata.get("quote_id")] or insight.supporting_quote_ids
            links.append(
                EvidenceLink(
                    insight_id=insight.insight_id,
                    insight_text=insight.text,
                    insight_type="objection",
                    supporting_quote_ids=supporting_ids,
                    confidence="multi_source" if len(set(supporting_ids)) > 1 else "single_source",
                )
            )
        for quote in quotes:
            if quote.category not in {"pain", "comparison"}:
                continue
            docs = retriever.invoke(quote.text)
            supporting_ids = [document.metadata["quote_id"] for document in docs if document.metadata.get("quote_id")] or [quote.quote_id]
            links.append(
                EvidenceLink(
                    insight_id=f"{quote.category}-{quote.quote_id}",
                    insight_text=quote.text,
                    insight_type=quote.category,
                    supporting_quote_ids=supporting_ids,
                    confidence="multi_source" if len(set(supporting_ids)) > 1 else "single_source",
                )
            )

        total_evidence_units = sum(len(link.supporting_quote_ids) for link in links)
        truth_density = total_evidence_units / max(1, len(links))
        synthesis.evidence_map = EvidenceMap(links=links, truth_density=truth_density, unsupported_insights=0)
        return {"synthesis": synthesis}

    def hitl_post_synthesis_node(state: ResearchState) -> dict:
        request = state["request"]
        if request.hitl_mode == "disabled":
            return {"hitl_post_synthesis": ReviewCheckpoint(status="auto_approved")}

        synthesis = state.get("synthesis") or SynthesisArtifacts()
        response = interrupt(
            {
                "stage": "post_synthesis_review",
                "truth_density": synthesis.evidence_map.truth_density,
                "insight_count": len(synthesis.evidence_map.links),
                "contradiction_count": len(synthesis.contradictions),
            }
        )
        approved = bool(response.get("approved", False))
        return {
            "hitl_post_synthesis": ReviewCheckpoint(
                status="approved" if approved else "rejected",
                notes=response.get("notes"),
            )
        }

    def validation_node(state: ResearchState) -> dict:
        current_thresholds = active_thresholds(state)
        failures: list[ValidationFailure] = []
        quotes = state.get("raw_quotes", [])
        synthesis = state.get("synthesis") or SynthesisArtifacts()
        categories = Counter(quote.category for quote in quotes)

        def require_count(actual: int, minimum: int, code: str, message: str, retry_instruction: str):
            if actual < minimum:
                failures.append(
                    ValidationFailure(
                        code=code,
                        message=f"{message}: found {actual}, expected at least {minimum}.",
                        severity="critical",
                        retry_instruction=retry_instruction,
                    )
                )

        require_count(categories["pain"], current_thresholds.min_pain_quotes, "insufficient_pain_quotes", "Pain quote coverage too low", "Generate stronger frustration and regret queries.")
        require_count(categories["failed_attempt"], current_thresholds.min_failed_attempts, "insufficient_failed_attempts", "Failed-attempt coverage too low", "Search for what people tried and why it failed.")
        require_count(categories["objection"], current_thresholds.min_objections, "insufficient_objections", "Objection coverage too low", "Search for pricing, trust, and adoption objections.")
        require_count(categories["switching_story"], current_thresholds.min_switching_stories, "insufficient_switching", "Switching story coverage too low", "Search for why people left competitors and what triggered the switch.")
        require_count(categories["comparison"], current_thresholds.min_comparisons, "insufficient_comparisons", "Comparison coverage too low", "Search for honest X vs Y comparisons.")
        require_count(categories["desire"], current_thresholds.min_desires, "insufficient_desires", "Desire coverage too low", "Search for quotes about what people wish existed.")

        if len(synthesis.contradictions) < current_thresholds.min_contradictions:
            failures.append(
                ValidationFailure(
                    code="missing_contradictions",
                    message="Synthesis did not find enough contradictions.",
                    severity="major",
                    retry_instruction="Look for opposing experiences across accepted quotes.",
                )
            )

        if len(synthesis.decision_moments) < current_thresholds.min_decision_moments:
            failures.append(
                ValidationFailure(
                    code="missing_decision_moments",
                    message="Decision moments are below threshold.",
                    severity="major",
                    retry_instruction="Collect more switching stories with concrete trigger moments.",
                )
            )

        if synthesis.evidence_map.truth_density < current_thresholds.min_truth_density:
            failures.append(
                ValidationFailure(
                    code="low_truth_density",
                    message="Truth density is below threshold.",
                    severity="critical",
                    retry_instruction="Collect more corroborating quotes before synthesis.",
                )
            )

        ratio, generic_ids, authenticity_failures, _ = quote_authenticity_metrics(quotes)
        if ratio > current_thresholds.max_generic_quote_ratio:
            failures.append(
                ValidationFailure(
                    code="generic_quote_ratio",
                    message=(
                        f"Generic or paraphrased quote ratio {ratio:.2%} exceeds "
                        f"the allowed {current_thresholds.max_generic_quote_ratio:.0%}."
                    ),
                    severity="critical",
                    retry_instruction=(
                        "Re-run query generation with stronger frustration/regret/switching/comparison queries "
                        "and recollect only weak evidence."
                    ),
                    related_quote_ids=generic_ids,
                )
            )

        supported_ids = {quote.quote_id for quote in quotes}
        orphan_ids = [
            quote_id
            for link in synthesis.evidence_map.links
            for quote_id in link.supporting_quote_ids
            if quote_id not in supported_ids
        ]
        if orphan_ids:
            failures.append(
                ValidationFailure(
                    code="orphan_evidence",
                    message="Evidence map references quote IDs that do not exist in accepted evidence.",
                    severity="critical",
                    related_quote_ids=orphan_ids,
                    retry_instruction="Rebuild evidence mapping from accepted quotes only.",
                )
            )

        status = "approved" if not failures else "retry"
        if failures and state.get("loop_count", 0) >= current_thresholds.max_loops:
            status = "escalated"

        return {
            "validation_failures": failures,
            "generic_quote_ratio": ratio,
            "generic_quote_ids": generic_ids,
            "quote_authenticity_failures": authenticity_failures,
            "retry_instructions": [failure.retry_instruction for failure in failures if failure.retry_instruction],
            "status": status,
            "loop_count": state.get("loop_count", 0) + (1 if failures and status == "retry" else 0),
        }

    def finalize_node(state: ResearchState) -> dict:
        synthesis = state.get("synthesis") or SynthesisArtifacts()
        packet = ApprovedResearchPacket(
            icp={
                "role": state["request"].audience,
                "company_size": state["request"].company_size or "unknown",
                "industry": state["request"].industry or "unknown",
                "pain_summary": state["request"].research_request,
            },
            raw_quotes=state.get("raw_quotes", []),
            objections=synthesis.objections,
            failed_attempts=synthesis.failed_attempts,
            decision_moments=synthesis.decision_moments,
            contradictions=synthesis.contradictions,
            voice_fingerprint=state.get("voice_fingerprint") or VoiceFingerprint(),
            competitors=state.get("competitors", []),
            evidence_map=synthesis.evidence_map,
            truth_density=synthesis.evidence_map.truth_density,
            evidence_coverage=1.0 if synthesis.evidence_map.links else 0.0,
            total_quotes=len(state.get("raw_quotes", [])),
            total_insights=len(synthesis.evidence_map.links),
            generic_quote_ratio=state.get("generic_quote_ratio", 0.0),
            generic_quote_ids=state.get("generic_quote_ids", []),
            quote_authenticity_failures=state.get("quote_authenticity_failures", []),
        )
        return {"final_research_packet": packet}

    def escalate_node(state: ResearchState) -> dict:
        synthesis = state.get("synthesis") or SynthesisArtifacts()
        partial = ApprovedResearchPacket(
            icp={
                "role": state["request"].audience,
                "company_size": state["request"].company_size or "unknown",
                "industry": state["request"].industry or "unknown",
                "pain_summary": state["request"].research_request,
            },
            raw_quotes=state.get("raw_quotes", []),
            objections=synthesis.objections,
            failed_attempts=synthesis.failed_attempts,
            decision_moments=synthesis.decision_moments,
            contradictions=synthesis.contradictions,
            voice_fingerprint=state.get("voice_fingerprint") or VoiceFingerprint(),
            competitors=state.get("competitors", []),
            evidence_map=synthesis.evidence_map,
            truth_density=synthesis.evidence_map.truth_density,
            evidence_coverage=1.0 if synthesis.evidence_map.links else 0.0,
            total_quotes=len(state.get("raw_quotes", [])),
            total_insights=len(synthesis.evidence_map.links),
            generic_quote_ratio=state.get("generic_quote_ratio", 0.0),
            generic_quote_ids=state.get("generic_quote_ids", []),
            quote_authenticity_failures=state.get("quote_authenticity_failures", []),
        )
        report = EscalationReport(
            partial_packet=partial,
            missing_items=[failure.message for failure in state.get("validation_failures", [])],
            attempted_queries=state.get("query_bank").executed_queries if state.get("query_bank") else [],
            truth_density=synthesis.evidence_map.truth_density,
            evidence_coverage=partial.evidence_coverage,
            validation_failures=state.get("validation_failures", []),
            recommendation="Manual review is needed to fill evidence gaps or improve quote authenticity.",
        )
        return {"escalation_report": report}

    builder = StateGraph(ResearchState)
    builder.add_node("strategist", strategist_node)
    builder.add_node("query_generation", query_generation_node)
    builder.add_node("query_quality_gate", query_quality_gate_node)
    builder.add_node("collect_voc", collect_voc_node)
    builder.add_node("collect_product", collect_product_node)
    builder.add_node("collect_competitors", collect_competitors_node)
    builder.add_node("quote_authenticity_gate", quote_authenticity_gate_node)
    builder.add_node("hitl_post_voc", hitl_post_voc_node)
    builder.add_node("voice_fingerprint", voice_fingerprint_node)
    builder.add_node("synthesis_stage_1", synthesis_stage_one_node)
    builder.add_node("synthesis_stage_2", synthesis_stage_two_node)
    builder.add_node("hitl_post_synthesis", hitl_post_synthesis_node)
    builder.add_node("validation", validation_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("escalate", escalate_node)
    builder.add_node("collect_fanout", lambda state: {})

    builder.add_edge(START, "strategist")
    builder.add_conditional_edges("strategist", lambda state: END if state.get("status") == "rejected" else "query_generation", [END, "query_generation"])
    builder.add_edge("query_generation", "query_quality_gate")
    builder.add_conditional_edges(
        "query_quality_gate",
        lambda state: "query_generation" if state.get("status") == "retry" else "escalate" if state.get("status") == "escalated" else "collect_fanout",
        ["query_generation", "escalate", "collect_fanout"],
    )
    builder.add_conditional_edges("collect_fanout", spawn_collection, ["collect_voc", "collect_product", "collect_competitors"])
    builder.add_edge("collect_voc", "quote_authenticity_gate")
    builder.add_edge("collect_product", "quote_authenticity_gate")
    builder.add_edge("collect_competitors", "quote_authenticity_gate")
    builder.add_edge("quote_authenticity_gate", "hitl_post_voc")
    builder.add_conditional_edges(
        "hitl_post_voc",
        lambda state: "escalate" if state.get("hitl_post_voc") and state["hitl_post_voc"].status == "rejected" else "voice_fingerprint",
        ["escalate", "voice_fingerprint"],
    )
    builder.add_edge("voice_fingerprint", "synthesis_stage_1")
    builder.add_edge("synthesis_stage_1", "synthesis_stage_2")
    builder.add_edge("synthesis_stage_2", "hitl_post_synthesis")
    builder.add_conditional_edges(
        "hitl_post_synthesis",
        lambda state: "escalate" if state.get("hitl_post_synthesis") and state["hitl_post_synthesis"].status == "rejected" else "validation",
        ["escalate", "validation"],
    )
    builder.add_conditional_edges(
        "validation",
        lambda state: "finalize" if state.get("status") == "approved" else "query_generation" if state.get("status") == "retry" else "escalate",
        ["finalize", "query_generation", "escalate"],
    )
    builder.add_edge("finalize", END)
    builder.add_edge("escalate", END)

    graph = builder.compile(checkpointer=checkpointer)

    def invoke(request, config=None):
        payload = _initial_state()
        payload["request"] = request
        return graph.invoke(payload, config=config)

    graph.invoke_request = invoke
    return graph
