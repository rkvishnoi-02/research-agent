"""Typed models and state for the research workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict


QuoteCategory = Literal[
    "pain",
    "failed_attempt",
    "objection",
    "switching_story",
    "comparison",
    "desire",
    "delight",
    "neutral",
]

SourceType = Literal[
    "reddit",
    "g2",
    "forum",
    "review_site",
    "hackernews",
    "web",
    "capterra",
    "trust_radius",
    "unknown",
]

QueryType = Literal["pain", "complaint", "switching", "comparison"]
IntentTarget = Literal["frustration", "regret", "switching", "comparison"]
ReviewStatus = Literal["auto_approved", "approved", "rejected"]
WorkflowStatus = Literal["pending", "running", "retry", "approved", "escalated", "rejected"]


class ResearchRequest(BaseModel):
    research_request: str
    product_name: str
    product_url: str | None = None
    audience: str
    industry: str | None = None
    company_size: str | None = None
    hitl_mode: Literal["enabled", "disabled"] = "disabled"
    focus_notes: str | None = None
    seed_quotes: list["RawQuote"] = Field(default_factory=list)
    seed_competitors: list[str] = Field(default_factory=list)


class RawQuote(BaseModel):
    """A verbatim buyer quote and its context."""

    quote_id: str = Field(default_factory=lambda: f"q-{uuid4().hex[:8]}")
    text: str
    source: str
    source_url: str
    source_type: SourceType
    context: str
    context_snippet: str
    category: QuoteCategory
    human_test_passed: bool = False
    human_test_failure_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_context(self) -> "RawQuote":
        if not self.context_snippet.strip():
            raise ValueError("context_snippet is required")
        return self


class QuoteAuditResult(BaseModel):
    quote_id: str
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    generic: bool = False


class ResearchQuery(BaseModel):
    query_text: str
    query_type: QueryType
    target_source: Literal["reddit", "g2", "web", "hn", "capterra"]
    expected_category: QuoteCategory
    intent_target: IntentTarget
    emotional_signal_score: float = 0.0
    rejection_reason: str | None = None


class QueryBank(BaseModel):
    pain_queries: list[ResearchQuery] = Field(default_factory=list)
    complaint_queries: list[ResearchQuery] = Field(default_factory=list)
    switching_queries: list[ResearchQuery] = Field(default_factory=list)
    comparison_queries: list[ResearchQuery] = Field(default_factory=list)
    executed_queries: list[str] = Field(default_factory=list)
    rejected_queries: list[ResearchQuery] = Field(default_factory=list)

    @property
    def all_queries(self) -> list[ResearchQuery]:
        return (
            self.pain_queries
            + self.complaint_queries
            + self.switching_queries
            + self.comparison_queries
        )


class CompetitorProfile(BaseModel):
    name: str
    positioning: str = ""
    primary_claims: list[str] = Field(default_factory=list)
    messaging_gaps: list[str] = Field(default_factory=list)
    buyer_perception_quote_ids: list[str] = Field(default_factory=list)


class VoiceFingerprint(BaseModel):
    category_language: list[str] = Field(default_factory=list)
    buyer_language: list[str] = Field(default_factory=list)
    fake_language: list[str] = Field(default_factory=list)
    emotional_triggers: list[str] = Field(default_factory=list)
    jargon_map: dict[str, str] = Field(default_factory=dict)


class FailedAttempt(BaseModel):
    what_they_tried: str
    why_it_failed: str
    supporting_quote_ids: list[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    quote_id_a: str
    quote_id_b: str
    statement_a_text: str
    statement_b_text: str
    tension: str
    signal_strength: Literal["weak", "moderate", "strong"]


class DecisionMoment(BaseModel):
    trigger: str
    decision: str
    supporting_quote_ids: list[str] = Field(default_factory=list)
    alternatives_considered: list[str] = Field(default_factory=list)


class Insight(BaseModel):
    insight_id: str
    text: str
    supporting_quote_ids: list[str] = Field(default_factory=list)


class EvidenceLink(BaseModel):
    insight_id: str
    insight_text: str
    insight_type: str
    supporting_quote_ids: list[str] = Field(default_factory=list)
    confidence: Literal["single_source", "multi_source", "cross_validated"] = "single_source"


class EvidenceMap(BaseModel):
    links: list[EvidenceLink] = Field(default_factory=list)
    truth_density: float = 0.0
    unsupported_insights: int = 0


class StageOneExtraction(BaseModel):
    extracted_pains: list[str] = Field(default_factory=list)
    extracted_desires: list[str] = Field(default_factory=list)
    extracted_objections: list[str] = Field(default_factory=list)
    extracted_switching_stories: list[str] = Field(default_factory=list)
    extracted_comparisons: list[str] = Field(default_factory=list)
    extracted_failed_attempts: list[str] = Field(default_factory=list)
    unclassified_quote_ids: list[str] = Field(default_factory=list)


class SynthesisArtifacts(BaseModel):
    objections: list[Insight] = Field(default_factory=list)
    failed_attempts: list[FailedAttempt] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    decision_moments: list[DecisionMoment] = Field(default_factory=list)
    evidence_map: EvidenceMap = Field(default_factory=EvidenceMap)


class ValidationFailure(BaseModel):
    code: str
    message: str
    severity: Literal["critical", "major", "minor"] = "critical"
    retry_instruction: str | None = None
    related_quote_ids: list[str] = Field(default_factory=list)


class ValidationThresholds(BaseModel):
    min_pain_quotes: int = 5
    min_failed_attempts: int = 3
    min_objections: int = 3
    min_switching_stories: int = 3
    min_comparisons: int = 3
    min_desires: int = 2
    min_contradictions: int = 2
    min_decision_moments: int = 3
    min_truth_density: float = 1.5
    max_generic_quote_ratio: float = 0.30
    max_loops: int = 3


class ApprovedResearchPacket(BaseModel):
    icp: dict[str, str]
    raw_quotes: list[RawQuote] = Field(default_factory=list)
    objections: list[Insight] = Field(default_factory=list)
    failed_attempts: list[FailedAttempt] = Field(default_factory=list)
    decision_moments: list[DecisionMoment] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    voice_fingerprint: VoiceFingerprint = Field(default_factory=VoiceFingerprint)
    competitors: list[CompetitorProfile] = Field(default_factory=list)
    evidence_map: EvidenceMap = Field(default_factory=EvidenceMap)
    truth_density: float = 0.0
    evidence_coverage: float = 1.0
    total_quotes: int = 0
    total_insights: int = 0
    generic_quote_ratio: float = 0.0
    generic_quote_ids: list[str] = Field(default_factory=list)
    quote_authenticity_failures: list[str] = Field(default_factory=list)


class EscalationReport(BaseModel):
    status: Literal["ESCALATED"] = "ESCALATED"
    partial_packet: ApprovedResearchPacket
    missing_items: list[str] = Field(default_factory=list)
    attempted_queries: list[str] = Field(default_factory=list)
    truth_density: float = 0.0
    evidence_coverage: float = 0.0
    validation_failures: list[ValidationFailure] = Field(default_factory=list)
    recommendation: str = ""


class ReviewCheckpoint(BaseModel):
    status: ReviewStatus = "auto_approved"
    notes: str | None = None


class ResearchState(TypedDict, total=False):
    request: ResearchRequest
    status: WorkflowStatus
    loop_count: int
    retry_instructions: list[str]
    query_bank: QueryBank | None
    candidate_quotes: Annotated[list[RawQuote], operator.add]
    raw_quotes: Annotated[list[RawQuote], operator.add]
    rejected_quotes: Annotated[list[RawQuote], operator.add]
    quote_audits: Annotated[list[QuoteAuditResult], operator.add]
    product_summary: str
    competitors: Annotated[list[CompetitorProfile], operator.add]
    voice_fingerprint: VoiceFingerprint | None
    stage_one: StageOneExtraction | None
    synthesis: SynthesisArtifacts | None
    validation_failures: list[ValidationFailure]
    generic_quote_ratio: float
    generic_quote_ids: list[str]
    quote_authenticity_failures: list[str]
    hitl_post_voc: ReviewCheckpoint | None
    hitl_post_synthesis: ReviewCheckpoint | None
    final_research_packet: ApprovedResearchPacket | None
    escalation_report: EscalationReport | None
    collection_logs: Annotated[list[str], operator.add]
