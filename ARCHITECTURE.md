# Deep Research Agent — System Architecture v2

> **Status:** DESIGN REVIEW — No code until approved  
> **Version:** 2.0 — refined with Query Generation Layer, Evidence Mapping, Split Synthesis, Truth Density, HITL  
> **Stack:** Python · LangChain · LangGraph · Pydantic  
> **Purpose:** Extract raw market truth for downstream content generation

---

## 1. System Architecture (Text Diagram)

```
                        ┌──────────────────┐
                        │  RESEARCH INPUT  │
                        │  (topic/product/ │
                        │   audience)      │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  STRATEGIST NODE │ ◄── Refuses vague inputs
                        │  Validates request│     Loops until clear
                        └────────┬─────────┘
                                 │ validated request
                        ┌────────▼─────────────┐
                        │  QUERY GENERATION    │ ◄── NEW: generates typed
                        │  LAYER               │     queries per category
                        │  (pain / complaint / │
                        │   switching / compare)│
                        └────────┬─────────────┘
                                 │ query_bank
            ┌────────────────────┼────────────────────┐
            │                    │                    │
   ┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
   │  VoC MINING     │ │ PRODUCT         │ │ COMPETITOR      │
   │  NODE (v2)      │ │ UNDERSTANDING   │ │ ANALYSIS NODE   │
   │  Per-category   │ │ NODE            │ │                 │
   │  quote minimums │ │                 │ │                 │
   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ 🔴 HITL CHECKPOINT #1  │ ◄── Human reviews raw
                    │ Post-VoC Review         │     data before synthesis
                    │ (optional, skippable)   │
                    └────────────┬────────────┘
                                 │
                        ┌────────▼─────────┐
                        │ VOICE            │
                        │ FINGERPRINT NODE │
                        └────────┬─────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │          SYNTHESIS (SPLIT)           │
              │                                      │
              │  ┌──────────────────────────────┐   │
              │  │ STAGE 1: RAW EXTRACTION      │   │
              │  │ Extract facts only.           │   │
              │  │ No interpretation.            │   │
              │  │ Assign quote_ids.             │   │
              │  └──────────────┬───────────────┘   │
              │                 │                    │
              │  ┌──────────────▼───────────────┐   │
              │  │ STAGE 2: PATTERN DETECTION   │   │
              │  │ Find contradictions.          │   │
              │  │ Map decision journeys.        │   │
              │  │ Build evidence_map.           │   │
              │  └──────────────┬───────────────┘   │
              └─────────────────┼────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ 🔴 HITL CHECKPOINT #2 │ ◄── Human reviews
                    │ Post-Synthesis Review   │     structured output
                    │ (optional, skippable)   │
                    └───────────┬────────────┘
                                │
                        ┌───────▼──────────┐
                  ┌────►│ VALIDATION NODE  │
                  │     │ + Truth Density  │
                  │     │ + Evidence Audit  │
                  │     └───────┬──────────┘
                  │             │
                  │    ┌────────▼─────────┐
                  │    │   PASS / FAIL?   │
                  │    └──┬──────────┬────┘
                  │       │          │
                  │   FAIL│      PASS│
                  │       │          │
                  │  ┌────▼───────┐  │
                  │  │Per-failure │  │
                  │  │retry with  │  │
                  │  │instructions│  │
                  │  └────┬───────┘  │
                  │       │          │
                  └───────┘ ┌────────▼─────────┐
                            │ APPROVED RESEARCH │
                            │ PACKET            │
                            └──────────────────┘
```

### Flow Summary (v2)

1. **Input** → Strategist validates → rejects if vague
2. **Query Generation** → Produces typed query bank (pain, complaint, switching, comparison)
3. **Parallel collection** → VoC (with query bank) + Product + Competitor run concurrently
4. **🔴 HITL #1** → Optional human review of raw data before synthesis
5. **Sequential enrichment** → Voice Fingerprint analyzes collected data
6. **Synthesis Stage 1** → Raw extraction with quote ID assignment (no interpretation)
7. **Synthesis Stage 2** → Pattern detection, contradiction finding, evidence mapping
8. **🔴 HITL #2** → Optional human review of structured output
9. **Validation** → Quality gate with truth density scoring + evidence audit
10. **Loop** → Failed validation routes back with per-failure retry instructions
11. **Output** → Approved Research Packet with full evidence map

### Changes from v1

| Component | v1 | v2 |
|-----------|----|----|
| Query generation | Implicit inside VoC | Dedicated node with typed categories |
| VoC collection | 15 quotes minimum total | Per-category minimums enforced |
| Evidence mapping | None | Every insight traced to quote IDs |
| Synthesis | Single node | Split: extraction → pattern detection |
| Validation | Threshold checks only | + Truth Density metric + evidence audit |
| HITL | Not supported | Two optional checkpoints |
| Retry logic | Generic re-run | Per-failure retry instructions |

---

## 2. State Schema (v2)

```python
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from uuid import uuid4


# ─────────────────────────────────────────────
# CORE DATA MODELS
# ─────────────────────────────────────────────

class RawQuote(BaseModel):
    """A single raw buyer quote with source and unique ID."""
    quote_id: str = Field(default_factory=lambda: f"q-{uuid4().hex[:8]}")
    text: str                          # exact buyer words, verbatim
    source: str                        # "r/saas", "G2 review", "HN comment"
    source_url: str                    # direct URL to original
    context: str                       # what prompted this statement
    category: Literal[
        "pain",             # frustrations, complaints
        "failed_attempt",   # things they tried that didn't work
        "objection",        # reasons they didn't buy / resisted
        "switching_story",  # why they moved from one solution to another
        "comparison",       # this vs that
        "desire",           # what they wish existed
        "delight",          # what they love about current solution
        "neutral"           # factual, no strong sentiment
    ]


class FailedAttempt(BaseModel):
    """Something the buyer tried and abandoned."""
    what_they_tried: str
    why_it_failed: str
    supporting_quote_ids: list[str]    # links back to RawQuote.quote_id


class Contradiction(BaseModel):
    """Two conflicting buyer statements."""
    quote_id_a: str                    # RawQuote.quote_id
    quote_id_b: str                    # RawQuote.quote_id
    statement_a_text: str              # quoted text for readability
    statement_b_text: str
    tension: str                       # what the conflict reveals
    signal_strength: Literal["weak", "moderate", "strong"]


class DecisionMoment(BaseModel):
    """The moment a buyer chose or rejected a solution."""
    trigger: str                       # what caused the decision
    decision: str                      # what they chose
    supporting_quote_ids: list[str]    # links back to RawQuote.quote_id
    alternatives_considered: list[str] = []


class CompetitorProfile(BaseModel):
    """Competitor messaging breakdown."""
    name: str
    positioning: str                   # how they describe themselves
    primary_claims: list[str]
    messaging_gaps: list[str]          # what they DON'T say
    buyer_perception_quote_ids: list[str]  # RawQuote IDs about this competitor


class VoiceFingerprint(BaseModel):
    """Language patterns detected in the market."""
    category_language: list[str]       # terms the category uses
    buyer_language: list[str]          # terms buyers actually use
    fake_language: list[str]           # marketing-speak buyers ignore
    emotional_triggers: list[str]      # phrases that drive action
    jargon_map: dict[str, str]         # marketing_term → buyer_term


# ─────────────────────────────────────────────
# QUERY GENERATION MODELS (NEW)
# ─────────────────────────────────────────────

class ResearchQuery(BaseModel):
    """A single search query with type and target source."""
    query_text: str
    query_type: Literal["pain", "complaint", "switching", "comparison"]
    target_source: Literal["reddit", "g2", "web", "hn", "capterra"]
    expected_category: str             # which RawQuote.category this should yield


class QueryBank(BaseModel):
    """Full set of queries generated before collection."""
    pain_queries: list[ResearchQuery]           # min 4
    complaint_queries: list[ResearchQuery]      # min 3
    switching_queries: list[ResearchQuery]      # min 3
    comparison_queries: list[ResearchQuery]     # min 3
    executed_queries: list[str] = []            # tracks what's been run (dedup)
    
    @property
    def total_queries(self) -> int:
        return (len(self.pain_queries) + len(self.complaint_queries) +
                len(self.switching_queries) + len(self.comparison_queries))
    
    @property
    def unexecuted(self) -> list[ResearchQuery]:
        """Return queries not yet run. Used on retry loops."""
        all_queries = (self.pain_queries + self.complaint_queries +
                      self.switching_queries + self.comparison_queries)
        return [q for q in all_queries if q.query_text not in self.executed_queries]


# ─────────────────────────────────────────────
# EVIDENCE MAPPING (NEW)
# ─────────────────────────────────────────────

class EvidenceLink(BaseModel):
    """Maps a single insight to its supporting evidence."""
    insight_id: str                    # unique ID for this insight
    insight_text: str                  # the insight statement
    insight_type: Literal[
        "pain", "desire", "objection", "pattern",
        "contradiction", "decision_trigger", "competitor_gap"
    ]
    supporting_quote_ids: list[str]    # must have >= 1 quote_id
    confidence: Literal["single_source", "multi_source", "cross_validated"]
    # single_source = 1 quote, multi_source = 2+ quotes, 
    # cross_validated = quotes from different sources agree


class EvidenceMap(BaseModel):
    """Complete mapping of all insights to supporting evidence."""
    links: list[EvidenceLink]
    
    @property
    def unsupported_insights(self) -> list[EvidenceLink]:
        """Any insight with 0 supporting quotes = violation."""
        return [link for link in self.links if len(link.supporting_quote_ids) == 0]
    
    @property
    def truth_density(self) -> float:
        """
        Truth Density = total evidence units / total insights
        
        evidence unit = one quote_id linked to one insight
        If we have 30 evidence links across 10 insights → density = 3.0
        
        Threshold: >= 1.5 to pass validation
        Target:    >= 2.5 for high-quality output
        """
        if not self.links:
            return 0.0
        total_evidence = sum(len(link.supporting_quote_ids) for link in self.links)
        return total_evidence / len(self.links)


# ─────────────────────────────────────────────
# SYNTHESIS STAGE OUTPUTS (NEW — split synthesis)
# ─────────────────────────────────────────────

class ExtractionOutput(BaseModel):
    """Stage 1 output: raw extraction, no interpretation."""
    extracted_pains: list[str]                    # quote_ids tagged "pain"
    extracted_desires: list[str]                  # quote_ids tagged "desire"
    extracted_objections: list[str]               # quote_ids tagged "objection"
    extracted_switching_stories: list[str]         # quote_ids tagged "switching_story"
    extracted_comparisons: list[str]              # quote_ids tagged "comparison"
    extracted_failed_attempts: list[str]          # quote_ids tagged "failed_attempt"
    unclassified_quote_ids: list[str]             # couldn't confidently classify


class PatternOutput(BaseModel):
    """Stage 2 output: patterns and contradictions found."""
    contradictions: list[Contradiction]
    decision_moments: list[DecisionMoment]
    failed_attempts: list[FailedAttempt]
    recurring_themes: list[dict]                  # {theme, frequency, quote_ids}
    evidence_map: EvidenceMap


# ─────────────────────────────────────────────
# HITL MODELS (NEW)
# ─────────────────────────────────────────────

class HITLCheckpoint(BaseModel):
    """Human-in-the-loop review point."""
    checkpoint_id: Literal["post_voc", "post_synthesis"]
    status: Literal["pending", "approved", "override", "skipped"]
    human_notes: str = ""              # free-form human feedback
    override_actions: list[str] = []   # e.g. ["remove_quote:q-abc123", "add_note:..."]
    reviewed_at: str | None = None     # ISO timestamp


# ─────────────────────────────────────────────
# VALIDATION MODELS (STRENGTHENED)
# ─────────────────────────────────────────────

class ValidationFailure(BaseModel):
    """A single validation failure with retry instructions."""
    check_name: str                    # e.g. "min_pain_quotes"
    severity: Literal["critical", "major", "minor"]
    current_value: int | float
    required_value: int | float
    retry_node: str                    # which node to re-run
    retry_instruction: str             # SPECIFIC instruction for the retry
    # Example: "Run 2 additional complaint queries on Reddit targeting 
    #           r/sales and r/startups. Current pain quotes: 2, need: 5."


class ValidationResult(BaseModel):
    """Complete validation output."""
    status: Literal["pass", "fail", "escalated"]
    failures: list[ValidationFailure]
    truth_density: float
    truth_density_pass: bool           # >= 1.5 threshold
    evidence_coverage: float           # % of insights with >= 1 quote
    unsupported_insight_count: int
    total_checks_run: int
    total_checks_passed: int


# ─────────────────────────────────────────────
# FINAL OUTPUT (UPGRADED)
# ─────────────────────────────────────────────

class ResearchPacket(BaseModel):
    """The final approved output — with full evidence traceability."""
    icp_summary: str
    
    # Raw evidence (all quotes, indexed by quote_id)
    raw_quotes: list[RawQuote]
    
    # Structured findings (all linked to quote_ids)
    failed_attempts: list[FailedAttempt]         # minimum 5
    objections: list[str]                         # minimum 5, each with quote_id ref
    decision_moments: list[DecisionMoment]        # minimum 3
    contradictions: list[Contradiction]           # minimum 2
    
    # Language
    category_language: list[str]
    voice_fingerprint: VoiceFingerprint
    
    # Evidence
    evidence_map: EvidenceMap                     # NEW: full insight→quote mapping
    proof_classification: dict[str, list[str]]    # "social_proof", "data_proof", etc.
    
    # Metrics
    truth_density: float                          # NEW: computed metric
    evidence_coverage: float                      # NEW: % insights supported


# ─────────────────────────────────────────────
# MAIN STATE (UPGRADED)
# ─────────────────────────────────────────────

class ResearchState(TypedDict):
    """LangGraph shared state — the single source of truth."""
    
    # --- Input ---
    research_request: str
    audience_definition: dict          # ICP details
    product_name: str
    product_url: str | None

    # --- Query Generation (NEW) ---
    query_bank: QueryBank | None

    # --- Raw collection (append-only) ---
    voc_raw_data: list[RawQuote]       # ALL quotes, indexed by quote_id
    product_data: dict                 # claims, features, positioning
    competitor_data: list[CompetitorProfile]
    voice_data: VoiceFingerprint | None

    # --- Synthesis outputs (NEW — split) ---
    extraction_output: ExtractionOutput | None     # Stage 1
    pattern_output: PatternOutput | None           # Stage 2
    evidence_map: EvidenceMap | None               # built in Stage 2

    # --- HITL (NEW) ---
    hitl_post_voc: HITLCheckpoint | None
    hitl_post_synthesis: HITLCheckpoint | None
    hitl_mode: Literal["enabled", "disabled"]      # global toggle

    # --- Processed (legacy compat) ---
    structured_insights: dict
    failed_attempts: list[FailedAttempt]
    contradictions: list[Contradiction]
    decision_moments: list[DecisionMoment]

    # --- Control ---
    validation_result: ValidationResult | None     # NEW: full result object
    validation_status: Literal["pending", "pass", "fail", "escalated"]
    validation_errors: list[ValidationFailure]     # NEW: typed failures
    loop_count: int                    # prevent infinite loops (max 3)
    retry_nodes: list[str]             # which nodes need re-run

    # --- Output ---
    final_research_packet: ResearchPacket | None
```

### State Rules (v2)

| Rule | Enforcement |
|------|-------------|
| `voc_raw_data` is append-only | New quotes are added, never overwritten |
| Every `RawQuote` gets a unique `quote_id` | Auto-generated on creation |
| `query_bank.executed_queries` tracks all run queries | Prevents duplicate queries on retry |
| `evidence_map` must exist before validation | Synthesis Stage 2 builds it |
| `loop_count` max = 3 | System escalates after 3 failed validation loops |
| `validation_errors` must include `retry_instruction` | No generic failures allowed |
| `final_research_packet` is None until validation passes | Quality gate is mandatory |
| HITL checkpoints default to "skipped" if `hitl_mode = "disabled"` | Autonomy is configurable |

---

## 3. Node Definitions (v2)

### 3.1 Strategist Node

| Attribute | Value |
|-----------|-------|
| **Input** | `research_request` (raw text) |
| **Output** | Validated `audience_definition`, clear `research_request` |
| **Responsibility** | Refuse vague inputs. Decompose request into structured fields |

**Validation Checklist:**
- [ ] Audience specified (who are they?)
- [ ] Goal specified (what do we want to learn?)
- [ ] Thesis specified (what do we believe going in?)
- [ ] Product/category specified
- [ ] At least one concrete question to answer

**Behavior on failure:** Returns `validation_errors` with what's missing. Does NOT proceed.

```
IF audience is vague → reject: "Specify role, company size, pain"
IF goal is "research everything" → reject: "Pick 1-2 specific angles"
IF thesis is missing → reject: "State your hypothesis"
```

---

### 3.2 Query Generation Layer (NEW)

| Attribute | Value |
|-----------|-------|
| **Input** | Validated `audience_definition`, `research_request`, `product_name` |
| **Output** | `query_bank` (QueryBank with typed queries per category) |
| **Responsibility** | Generate diverse, high-signal search queries. Prevent duplicate patterns |

**Must generate queries across 4 mandatory categories:**

| Category | Min Queries | Query Pattern | Expected Output |
|----------|-------------|---------------|-----------------|
| **Pain** | 4 | Emotional language, frustration-first | `RawQuote.category = "pain"` |
| **Complaint** | 3 | Product-specific grievances, failure stories | `RawQuote.category = "failed_attempt"` |
| **Switching** | 3 | "Why I left X", migration stories | `RawQuote.category = "switching_story"` |
| **Comparison** | 3 | "X vs Y", head-to-head evaluations | `RawQuote.category = "comparison"` |

**Query diversity rules:**

```
RULE 1: No two queries may share > 60% of the same keywords
RULE 2: Queries must target at least 3 different sources
         (e.g., Reddit, G2, web — not all Reddit)
RULE 3: Each category must have at least 1 query targeting 
         negative sentiment specifically
RULE 4: Comparison queries must name specific competitor products
```

**Example query bank for "sales engagement platforms for SDRs":**

```yaml
pain_queries:
  - query: "sales engagement tool too complicated SDR frustrating"
    target: reddit
  - query: "SDR team won't use outreach tool wasting time"
    target: reddit
  - query: "sales automation setup takes forever small team"
    target: web
  - query: "Outreach.io Salesloft 1-star reviews"
    target: g2

complaint_queries:
  - query: "worst thing about Apollo.io sales"
    target: reddit
  - query: "sales engagement software bugs problems"
    target: g2
  - query: "SDR tool that doesn't actually work"
    target: web

switching_queries:
  - query: "why I switched from Salesloft to something else"
    target: reddit
  - query: "leaving Outreach.io what we moved to"
    target: reddit
  - query: "migrated from Apollo honest experience SDR"
    target: web

comparison_queries:
  - query: "Salesloft vs Outreach vs Apollo which is better SDR"
    target: reddit
  - query: "Outreach.io alternative for small sales team"
    target: g2
  - query: "comparing sales engagement platforms 2025"
    target: web
```

**On retry (validation failed):**
- Receives `validation_errors` with specific gaps
- Generates ONLY new queries targeting the gaps
- Marks gap-filling queries with `is_retry = True`
- Cannot reuse any query in `query_bank.executed_queries`

---

### 3.3 VoC Mining Node v2 (CRITICAL — UPGRADED)

| Attribute | Value |
|-----------|-------|
| **Input** | `query_bank`, `audience_definition` |
| **Output** | Appends to `voc_raw_data` (list of `RawQuote` with `quote_id` and `category`) |
| **Responsibility** | Execute queries from bank. Collect raw buyer language. Enforce per-category minimums |

**Per-Category Minimums (ALL must be met):**

| Category | Minimum Quotes | Source |
|----------|---------------|--------|
| `pain` | 5 | Pain queries |
| `failed_attempt` | 3 | Complaint queries |
| `objection` | 3 | Any source |
| `switching_story` | 3 | Switching queries |
| `comparison` | 3 | Comparison queries |
| `desire` | 2 | Any source |

**Total minimum: 19 quotes** (up from 15 in v1)

**Hard Rules (unchanged + new):**
- Every output must be a direct quote with source URL
- NO paraphrasing. NO "customers feel that..."
- Must assign `category` to each quote on collection
- Must assign unique `quote_id` on creation
- Must mark each query as executed in `query_bank.executed_queries`
- If a query returns 0 relevant results → log it, don't count as failure

**Execution order:**
1. Execute all `pain_queries` first (highest priority)
2. Execute `complaint_queries`
3. Execute `switching_queries`
4. Execute `comparison_queries`
5. After all queries: check per-category minimums
6. If any category below minimum → flag for Query Generation retry

**Re-run behavior (on validation failure):**
- Receives `retry_instruction` with exact gaps (e.g., "Need 2 more switching_story quotes")
- Query Generation Layer generates targeted gap-filling queries
- VoC executes only the new queries
- Appends new quotes (existing quotes preserved)

---

### 3.4 Product Understanding Node

*(Unchanged from v1 — see original architecture)*

| Attribute | Value |
|-----------|-------|
| **Input** | `product_name`, `product_url` |
| **Output** | `product_data` dict |
| **Responsibility** | Extract what the product claims vs. reality |

**Extracts:**
- Official positioning statement
- Feature claims (what they say they do)
- Pricing model
- Target audience (as stated by the product)
- Messaging patterns (words/phrases used repeatedly)

**Critical comparison:** Maps product claims against `voc_raw_data` to find perception gaps:
- Claims buyers confirm? → `validated_claims`
- Claims buyers dispute? → `disputed_claims`
- Claims buyers don't mention? → `invisible_claims`

---

### 3.5 Competitor Analysis Node

*(Unchanged from v1 — see original architecture)*

| Attribute | Value |
|-----------|-------|
| **Input** | `audience_definition`, `product_name` |
| **Output** | `competitor_data` (list of `CompetitorProfile`) |
| **Responsibility** | Map competitor messaging landscape |

**For each competitor (up to 5):**
1. Extract homepage/landing page positioning
2. Pull ad messaging (if available)
3. Collect buyer reviews about the competitor
4. Identify messaging patterns across competitors

**Output structure:**
- What ALL competitors say (table stakes)
- What NOBODY says (opportunity gaps)
- What buyers say about competitors (raw quotes with `quote_id`)
- Weakness patterns (where competitors fail)

---

### 3.6 HITL Checkpoint #1: Post-VoC Review (NEW)

| Attribute | Value |
|-----------|-------|
| **Input** | All `voc_raw_data`, `product_data`, `competitor_data` |
| **Output** | `hitl_post_voc` (HITLCheckpoint) |
| **Responsibility** | Pause for optional human review of raw collected data |

**When `hitl_mode = "enabled"`:**
1. System presents summary of collected data:
   - Quote count per category
   - Source distribution
   - Category coverage gaps
   - Top 5 highest-engagement quotes
2. Human can:
   - **Approve** → proceed to Voice Fingerprint
   - **Override** → remove bad quotes, add notes, redirect focus
   - **Skip** → treated as approve

**When `hitl_mode = "disabled"`:**
- Checkpoint status set to "skipped"
- Flow continues automatically

**Human override actions supported:**
```
remove_quote:<quote_id>        # remove a quote that seems inauthentic
add_note:<text>                # add context for synthesis to consider
redirect_focus:<instruction>   # e.g. "focus more on pricing objections"
approve_with_gaps              # proceed even if minimums not fully met
```

---

### 3.7 Voice Fingerprint Node

*(Unchanged from v1 — see original architecture)*

| Attribute | Value |
|-----------|-------|
| **Input** | All `voc_raw_data`, `competitor_data` |
| **Output** | `voice_data` (`VoiceFingerprint`) |
| **Responsibility** | Detect language patterns and authenticity markers |

**Process:**
1. Analyze all collected raw quotes
2. Separate buyer language from marketing language
3. Identify emotional trigger phrases
4. Build jargon map (marketing term → buyer term)
5. Flag "fake" language (words buyers never use)

---

### 3.8 Synthesis Stage 1: Raw Extraction (NEW — was part of Synthesis Node)

| Attribute | Value |
|-----------|-------|
| **Input** | `voc_raw_data`, `product_data`, `competitor_data`, `voice_data` |
| **Output** | `extraction_output` (ExtractionOutput) |
| **Responsibility** | ONLY extract and classify. Zero interpretation. Zero pattern-finding |

**What Stage 1 does:**
1. Take every `RawQuote` in `voc_raw_data`
2. Verify its `category` tag is correct (re-classify if needed)
3. Group quotes by category into extraction lists
4. Flag any quotes that couldn't be confidently classified → `unclassified_quote_ids`
5. Output clean lists of quote_ids per category

**What Stage 1 does NOT do:**
- ❌ No pattern detection
- ❌ No "insights" or "themes"
- ❌ No contradiction finding
- ❌ No narrative construction
- ❌ No summarization of any kind

**Hard rule:** The output of Stage 1 must be verifiable by comparing quote_ids against `voc_raw_data`. Any quote_id that doesn't exist in state → ERROR.

---

### 3.9 Synthesis Stage 2: Pattern & Contradiction Detection (NEW — was part of Synthesis Node)

| Attribute | Value |
|-----------|-------|
| **Input** | `extraction_output`, `voc_raw_data` (for full quote text), `voice_data` |
| **Output** | `pattern_output` (PatternOutput) with `evidence_map` |
| **Responsibility** | Find patterns, contradictions, and decision journeys. Build evidence map |

**What Stage 2 does:**

**Step 1 — Contradiction Detection:**
```
FOR each pair of quotes in the same topic area:
  IF they express opposing views:
    CREATE Contradiction(
      quote_id_a, quote_id_b,
      tension = what the conflict reveals,
      signal_strength = weak|moderate|strong
    )

Signal strength:
  weak     = same person, different contexts
  moderate = different people, same topic
  strong   = different people, opposite conclusions, both high-engagement
```

**Step 2 — Decision Journey Mapping:**
```
FOR each switching_story quote:
  EXTRACT:
    trigger    = what caused the decision
    decision   = what they chose
    alternatives = what else they considered
  LINK to supporting_quote_ids
```

**Step 3 — Failed Attempt Extraction:**
```
FOR each failed_attempt quote:
  EXTRACT:
    what_they_tried = the solution/approach
    why_it_failed   = the reason
  LINK to supporting_quote_ids
```

**Step 4 — Evidence Map Construction:**
```
FOR each insight/pattern found:
  CREATE EvidenceLink(
    insight_id   = unique ID
    insight_text = the insight statement
    insight_type = pain|desire|objection|pattern|...
    supporting_quote_ids = [all quote_ids that support this]
    confidence = single_source|multi_source|cross_validated
  )

RULE: If supporting_quote_ids is empty → DO NOT create the insight.
      It doesn't exist without evidence.
```

**Step 5 — Recurring Theme Detection:**
```
FOR quotes with similar content across different sources:
  GROUP into themes with:
    theme     = description
    frequency = how many quotes support it
    quote_ids = all supporting quotes
  
  ONLY report themes with frequency >= 3
```

---

### 3.10 HITL Checkpoint #2: Post-Synthesis Review (NEW)

| Attribute | Value |
|-----------|-------|
| **Input** | `pattern_output`, `evidence_map`, `voc_raw_data` |
| **Output** | `hitl_post_synthesis` (HITLCheckpoint) |
| **Responsibility** | Pause for optional human review of structured output before validation |

**When `hitl_mode = "enabled"`:**
1. System presents:
   - All contradictions found (with signal strength)
   - Decision moments mapped
   - Evidence map summary (truth density score)
   - Any unclassified quotes
   - Insights with only single-source confidence
2. Human can:
   - **Approve** → proceed to validation
   - **Override** → reclassify quotes, merge/split contradictions, add missing context
   - **Skip** → treated as approve

**Human override actions supported:**
```
reclassify_quote:<quote_id>:<new_category>
merge_contradictions:<id_a>:<id_b>
split_insight:<insight_id>
add_missing_context:<insight_id>:<text>
flag_as_invented:<insight_id>         # human identifies hallucination
force_rerun_voc:<instruction>         # human requests more data
```

---

### 3.11 Validation Node (STRENGTHENED)

| Attribute | Value |
|-----------|-------|
| **Input** | Full state after synthesis (both stages) + evidence_map |
| **Output** | `validation_result` (ValidationResult), `validation_status`, `retry_nodes` |
| **Responsibility** | Quality gate with truth density scoring, evidence audit, and per-failure retry instructions |

**Three validation layers (ALL must pass):**

#### Layer 1: Category Coverage

| Check | Threshold | Severity | Retry Node | Retry Instruction Template |
|-------|-----------|----------|------------|---------------------------|
| Pain quotes | ≥ 5 | critical | query_generation → voc_mining | "Generate {deficit} pain queries targeting {suggested_sources}. Focus on: {suggested_angles}" |
| Failed attempt quotes | ≥ 3 | critical | query_generation → voc_mining | "Generate complaint queries targeting negative reviews on G2 for {competitors}" |
| Objection quotes | ≥ 3 | major | query_generation → voc_mining | "Search for 'why I didn't buy {product}' and 'reasons not to use {category}' on Reddit" |
| Switching story quotes | ≥ 3 | major | query_generation → voc_mining | "Search for 'switched from {competitor}' and 'why I left {competitor}' stories" |
| Comparison quotes | ≥ 3 | major | query_generation → voc_mining | "Run comparison queries: '{comp_a} vs {comp_b} honest review'" |
| Desire quotes | ≥ 2 | minor | query_generation → voc_mining | "Search for 'I wish {category} had' and 'ideal {category} would'" |
| Contradictions | ≥ 2 | critical | synthesis_stage_2 | "Re-analyze quotes in {low_coverage_categories}. Look for opposing views on {topics}" |
| Decision moments | ≥ 3 | major | synthesis_stage_2 | "Re-examine switching_story quotes for decision triggers" |
| Failed attempts (structured) | ≥ 5 | critical | synthesis_stage_2 | "Map failed_attempt quotes into structured FailedAttempt objects" |
| Competitor profiles | ≥ 2 | major | competitor_analysis | "Expand search to include {suggested_competitors}" |
| Voice fingerprint | all fields filled | major | voice_fingerprint | "Re-run with current quote set" |

#### Layer 2: Evidence Audit (NEW)

| Check | Rule | On Fail |
|-------|------|---------|
| Unsupported insights | Must = 0 | Strip ALL insights with `supporting_quote_ids = []` |
| Orphan quote_ids | All referenced IDs must exist in `voc_raw_data` | Remove broken references |
| Single-source insights | Flag, don't reject | Add warning to final packet |
| Invented content | Any insight text not derivable from quotes | HARD REJECT, strip the insight |

**Evidence audit process:**
```python
def audit_evidence(state: ResearchState) -> list[ValidationFailure]:
    failures = []
    evidence_map = state["evidence_map"]
    quote_ids_in_state = {q.quote_id for q in state["voc_raw_data"]}
    
    for link in evidence_map.links:
        # Check: every quote_id must exist
        for qid in link.supporting_quote_ids:
            if qid not in quote_ids_in_state:
                link.supporting_quote_ids.remove(qid)  # auto-fix
        
        # Check: no insight can have 0 evidence after cleanup
        if len(link.supporting_quote_ids) == 0:
            failures.append(ValidationFailure(
                check_name="unsupported_insight",
                severity="critical",
                current_value=0,
                required_value=1,
                retry_node="synthesis_stage_2",
                retry_instruction=f"Insight '{link.insight_text}' has no "
                    f"supporting evidence. Either find a supporting quote "
                    f"or remove this insight entirely."
            ))
    
    return failures
```

#### Layer 3: Truth Density (NEW)

```python
def check_truth_density(evidence_map: EvidenceMap) -> ValidationFailure | None:
    """
    Truth Density = total evidence units / total insights
    
    evidence_unit = one quote_id supporting one insight
    
    Example:
      10 insights, each supported by 3 quotes → density = 3.0 ✓
      10 insights, each supported by 1 quote  → density = 1.0 ✗
      10 insights, 3 have 0 quotes            → HARD FAIL (unsupported)
    
    Thresholds:
      >= 2.5  EXCELLENT — high confidence output
      >= 1.5  PASS      — minimum acceptable
      <  1.5  FAIL      — insufficient evidence backing
    """
    density = evidence_map.truth_density
    
    if density < 1.5:
        return ValidationFailure(
            check_name="truth_density",
            severity="critical",
            current_value=density,
            required_value=1.5,
            retry_node="voc_mining",
            retry_instruction=f"Truth density is {density:.2f} (need >= 1.5). "
                f"Collect more supporting evidence for existing insights. "
                f"Focus on insights currently backed by only 1 quote. "
                f"Target: find 2nd/3rd source for each single-source insight."
        )
    return None
```

**Full validation flow:**
```
1. Run Layer 1: Category Coverage → collect failures
2. Run Layer 2: Evidence Audit → collect failures + auto-strip unsupported
3. Run Layer 3: Truth Density → collect failures
4. IF any critical failures → status = "fail", set retry_nodes
5. IF only minor failures → status = "pass" with warnings
6. IF loop_count >= 3 → status = "escalated"
7. Build ValidationResult with complete details
```

**Rejection examples:**

```
REJECT: Insight "Teams prefer simple tools" with 0 supporting quotes
  → retry_instruction: "Remove this insight. It has no evidence."

REJECT: truth_density = 1.1 (below 1.5 threshold)
  → retry_instruction: "15 insights but only 16 evidence units. 
     Need at least 23 evidence units. Focus on finding additional 
     supporting quotes for the 8 single-source insights."

REJECT: 0 switching_story quotes
  → retry_instruction: "Generate 3 switching queries: 'why I left 
     {competitor_a}', 'switched from {competitor_b} to what', 
     'migrating from {competitor_c} experience'. Execute on Reddit."

REJECT: Contradiction signal_strength all "weak"
  → retry_instruction: "Current contradictions are from same 
     person/context. Find contradictions from DIFFERENT buyers 
     reaching OPPOSITE conclusions about same product/feature."
```

---

## 4. Edge Logic (v2 — Flow Control)

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

graph = StateGraph(ResearchState)
memory = MemorySaver()  # enables HITL interrupt/resume

# --- Add nodes ---
graph.add_node("strategist", strategist_node)
graph.add_node("query_generation", query_generation_node)          # NEW
graph.add_node("voc_mining", voc_mining_node)
graph.add_node("product_understanding", product_understanding_node)
graph.add_node("competitor_analysis", competitor_analysis_node)
graph.add_node("hitl_post_voc", hitl_post_voc_node)                # NEW
graph.add_node("voice_fingerprint", voice_fingerprint_node)
graph.add_node("synthesis_stage_1", synthesis_extraction_node)     # NEW (split)
graph.add_node("synthesis_stage_2", synthesis_pattern_node)        # NEW (split)
graph.add_node("hitl_post_synthesis", hitl_post_synthesis_node)    # NEW
graph.add_node("validation", validation_node)

# --- Entry ---
graph.set_entry_point("strategist")

# --- Strategist → Query Generation ---
graph.add_edge("strategist", "query_generation")

# --- Query Generation → Parallel collection ---
graph.add_edge("query_generation", "voc_mining")
graph.add_edge("query_generation", "product_understanding")
graph.add_edge("query_generation", "competitor_analysis")

# --- Collection → HITL #1 (fan-in) ---
graph.add_edge("voc_mining", "hitl_post_voc")
graph.add_edge("product_understanding", "hitl_post_voc")
graph.add_edge("competitor_analysis", "hitl_post_voc")

# --- HITL #1 → conditional ---
def route_after_hitl_voc(state: ResearchState) -> str:
    checkpoint = state.get("hitl_post_voc")
    if checkpoint and checkpoint.status == "override":
        # Human requested re-run with modifications
        if "force_rerun_voc" in str(checkpoint.override_actions):
            return "query_generation"
    return "voice_fingerprint"

graph.add_conditional_edges("hitl_post_voc", route_after_hitl_voc)

# --- Voice Fingerprint → Synthesis Stage 1 ---
graph.add_edge("voice_fingerprint", "synthesis_stage_1")

# --- Synthesis Stage 1 → Stage 2 ---
graph.add_edge("synthesis_stage_1", "synthesis_stage_2")

# --- Synthesis Stage 2 → HITL #2 ---
graph.add_edge("synthesis_stage_2", "hitl_post_synthesis")

# --- HITL #2 → conditional ---
def route_after_hitl_synthesis(state: ResearchState) -> str:
    checkpoint = state.get("hitl_post_synthesis")
    if checkpoint and checkpoint.status == "override":
        if "force_rerun_voc" in str(checkpoint.override_actions):
            return "query_generation"
    return "validation"

graph.add_conditional_edges("hitl_post_synthesis", route_after_hitl_synthesis)

# --- Validation → Conditional routing ---
def route_after_validation(state: ResearchState) -> str:
    if state["validation_status"] == "pass":
        return END
    if state["validation_status"] == "escalated":
        return END
    if state["loop_count"] >= 3:
        return END  # force escalation
    
    retry = state["retry_nodes"]
    
    # Priority-ordered retry routing
    if "query_generation" in retry or "voc_mining" in retry:
        return "query_generation"  # always re-gen queries before re-mining
    if "synthesis_stage_2" in retry:
        return "synthesis_stage_2"
    if "synthesis_stage_1" in retry:
        return "synthesis_stage_1"
    if "competitor_analysis" in retry:
        return "competitor_analysis"
    if "voice_fingerprint" in retry:
        return "voice_fingerprint"
    return END

graph.add_conditional_edges("validation", route_after_validation)

# --- Compile with checkpointer for HITL ---
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["hitl_post_voc", "hitl_post_synthesis"]
    # LangGraph will pause here and wait for human input
    # when hitl_mode = "enabled"
)
```

---

## 5. Loop Logic (v2)

```
LOOP 1 (most common): Category coverage failure
  validation → query_generation (gap-filling queries) → voc_mining → 
  hitl_post_voc → voice_fingerprint → synthesis_stage_1 → 
  synthesis_stage_2 → hitl_post_synthesis → validation

LOOP 2: Truth density failure (need more evidence for existing insights)
  validation → query_generation (targeted queries) → voc_mining → 
  hitl_post_voc → voice_fingerprint → synthesis_stage_1 → 
  synthesis_stage_2 → hitl_post_synthesis → validation

LOOP 3: Pattern detection failure (contradictions/decisions missing)
  validation → synthesis_stage_2 (re-analyze existing data) → 
  hitl_post_synthesis → validation

LOOP 4: Competitor data insufficient  
  validation → competitor_analysis → hitl_post_voc → 
  voice_fingerprint → synthesis_stage_1 → synthesis_stage_2 → 
  hitl_post_synthesis → validation

MAX LOOPS: 3
AFTER MAX: System outputs partial packet as "ESCALATED" with:
  - What passed
  - What failed (with exact gaps)
  - All attempted queries
  - Recommendations for manual research
```

### Loop Safeguards (v2)

| Safeguard | Implementation |
|-----------|---------------|
| Max retry count | `loop_count` increments each cycle, halts at 3 |
| No duplicate queries | `query_bank.executed_queries` tracks all run queries |
| Targeted re-queries | Query Gen receives `validation_errors` with specific retry instructions |
| Gap-only collection | On retry, VoC only runs new queries (doesn't re-run passed ones) |
| Partial results preserved | Append-only state means good data is never lost |
| Evidence map rebuild | Stage 2 rebuilds `evidence_map` from full state each time |
| HITL on retry | Checkpoints still fire on retry loops (human can course-correct) |

---

## 6. Tool Definitions

*(Tools unchanged from v1 — see original architecture for full definitions)*

| Tool | Purpose | Key Provider |
|------|---------|-------------|
| **WebSearchTool** | General web search | Tavily / SerpAPI |
| **RedditSearchTool** | Semantic Reddit search | reddit-insights MCP / PRAW |
| **ReviewScrapingTool** | G2, Capterra, TrustRadius reviews | Apify actors |
| **ContentExtractionTool** | Clean text from URLs | Jina Reader / Firecrawl |
| **CompetitorAdTool** | Ad library extraction | Facebook Ad Library API |

---

## 7. Example Execution Flow (v2)

### Scenario: Research for a B2B SaaS sales engagement platform

**Input:**
```
research_request: "Research buyer language and pain points for sales 
                   engagement platforms targeting SDR teams at mid-market 
                   SaaS companies (100-500 employees)"
product_name: "Synvo"
product_url: "https://synvo.me"
hitl_mode: "enabled"
```

**Step 1 — Strategist Node**
```
✓ Audience: SDR teams at mid-market SaaS (100-500 emp)
✓ Goal: Buyer language + pain points for sales engagement
✓ Thesis: SDR teams are frustrated with complex tools 
          that require admin overhead
→ PASS
```

**Step 2 — Query Generation Layer (NEW)**
```
Generated query bank:
  Pain (4):
    "sales engagement tool too complicated SDR" → reddit
    "SDR team won't use outreach tool" → reddit
    "sales automation setup takes forever" → web
    "Outreach.io frustrating 1-star" → g2
  
  Complaint (3):
    "worst thing about Apollo.io" → reddit
    "sales engagement software bugs" → g2
    "SDR tool that doesn't work" → web
  
  Switching (3):
    "why I switched from Salesloft" → reddit
    "leaving Outreach what we moved to" → reddit
    "migrated from Apollo experience" → web
  
  Comparison (3):
    "Salesloft vs Outreach vs Apollo SDR" → reddit
    "Outreach alternative small team" → g2
    "sales engagement platforms compared 2025" → web

Total: 13 queries, 3 sources, 0 duplicates ✓
```

**Step 3 — VoC Mining v2 (parallel with Steps 4-5)**
```
Executed 13 queries from bank.
Collection by category:
  pain:             7 quotes (min: 5) ✓
  failed_attempt:   4 quotes (min: 3) ✓
  objection:        5 quotes (min: 3) ✓
  switching_story:  4 quotes (min: 3) ✓
  comparison:       3 quotes (min: 3) ✓
  desire:           3 quotes (min: 2) ✓

Total: 26 quotes, all with quote_ids assigned
All queries marked as executed in bank

Example quote:
  quote_id: q-3f8a1bc2
  text: "We spent 3 months setting up sequences and 
         half our SDRs still use Gmail"
  source: r/sales
  source_url: https://reddit.com/r/sales/comments/...
  category: pain
```

**Step 4 — Product Understanding (parallel)**
```
(unchanged from v1)
```

**Step 5 — Competitor Analysis (parallel)**
```
Competitors: Salesloft, Outreach, Apollo.io
All buyer perception quotes now linked by quote_id
```

**Step 6 — 🔴 HITL Checkpoint #1: Post-VoC Review**
```
System presents:
  "26 quotes collected across 6 categories. 
   All minimums met. 3 sources used.
   Top engagement quote: q-3f8a1bc2 (+47 upvotes)
   
   Proceed to synthesis?"

Human response: "Approved. Focus synthesis on pricing objections."
→ hitl_post_voc.status = "approved"
→ hitl_post_voc.human_notes = "Focus synthesis on pricing objections"
```

**Step 7 — Voice Fingerprint**
```
Category language: "sales engagement", "sequences", "cadences"
Buyer language: "follow-up system", "outreach tool", "email sender"
Fake language: "revenue intelligence platform", "go-to-market suite"
Emotional triggers: "I just want it to work", "I don't have time to configure"
```

**Step 8 — Synthesis Stage 1: Raw Extraction**
```
Extraction output (no interpretation, just classification):
  extracted_pains: [q-3f8a1bc2, q-9a2c4df1, q-7b3e5af0, ...]  (7 IDs)
  extracted_desires: [q-1c4d6be2, q-8f2a3ce1, q-5d7e9ab3]     (3 IDs)
  extracted_objections: [q-2d5e7cf3, q-6a8b0de4, ...]          (5 IDs)
  extracted_switching_stories: [q-4e6f8ag4, q-0b2c4ef5, ...]   (4 IDs)
  extracted_comparisons: [q-3a5b7dg5, q-7c9d1eh6, q-1e3f5fi7]  (3 IDs)
  extracted_failed_attempts: [q-5f7g9bh6, q-9d1e3gj7, ...]     (4 IDs)
  unclassified_quote_ids: []                                     (0)

All quote_ids verified against voc_raw_data ✓
```

**Step 9 — Synthesis Stage 2: Pattern Detection**
```
Contradictions found: 3
  1. q-3f8a1bc2 vs q-7c9d1eh6 (STRONG)
     "Setup takes forever" vs "Once set up, it's the best tool we've used"
     Tension: Complexity is tolerable IF onboarding is supported

  2. q-9a2c4df1 vs q-1c4d6be2 (MODERATE)
     "Too expensive for what it does" vs "Worth every penny for the time saved"
     Tension: Value perception depends on team size

  3. q-2d5e7cf3 vs q-8f2a3ce1 (MODERATE)
     "AI features are gimmicky" vs "AI personalization is the killer feature"
     Tension: AI maturity varies by vendor

Decision moments: 4 (✓ >3)
Failed attempts (structured): 6 (✓ >5)

Evidence map built:
  Total insights: 18
  Total evidence units: 52
  Truth density: 2.89 (✓ EXCELLENT, threshold: 1.5)
  Unsupported insights: 0 (✓)
  Cross-validated insights: 7
  Single-source insights: 3 (flagged, not rejected)
```

**Step 10 — 🔴 HITL Checkpoint #2: Post-Synthesis Review**
```
System presents:
  "18 insights mapped to 52 evidence units.
   Truth density: 2.89 (excellent).
   3 contradictions found (1 strong, 2 moderate).
   3 single-source insights flagged for review.
   
   Review and approve?"

Human response: "Approved. Good quality."
→ hitl_post_synthesis.status = "approved"
```

**Step 11 — Validation**
```
Layer 1: Category Coverage
  ✓ pain: 7 (min 5)
  ✓ failed_attempt: 4 (min 3)
  ✓ objection: 5 (min 3)
  ✓ switching_story: 4 (min 3)
  ✓ comparison: 3 (min 3)
  ✓ desire: 3 (min 2)
  ✓ contradictions: 3 (min 2)
  ✓ decision_moments: 4 (min 3)
  ✓ failed_attempts_structured: 6 (min 5)
  ✓ competitors: 3 (min 2)
  ✓ voice_fingerprint: complete

Layer 2: Evidence Audit
  ✓ Unsupported insights: 0
  ✓ Orphan quote_ids: 0
  ⚠ Single-source insights: 3 (warning, not failure)

Layer 3: Truth Density
  ✓ truth_density: 2.89 (threshold: 1.5)

→ PASS: All layers passed. 
→ Output: Approved Research Packet with full evidence map
```

---

## 8. Failure Handling Logic (v2)

### Failure Matrix (upgraded with retry instructions)

| Failure Type | Detection | Severity | Recovery | Retry Instruction |
|-------------|-----------|----------|----------|-------------------|
| Vague input | Strategist checks | critical | Reject with specific missing fields | N/A (user must fix) |
| Insufficient pain quotes | Validation Layer 1: <5 | critical | Re-run Query Gen → VoC | "Generate {N} pain queries for {sources}" |
| Missing switching stories | Validation Layer 1: <3 | major | Re-run Query Gen → VoC | "Search 'why I left {competitor}'" |
| No contradictions | Validation Layer 1: <2 | critical | Re-run Synthesis Stage 2 | "Look for opposing views in {categories}" |
| Weak contradictions | All signal_strength="weak" | major | Re-run Synthesis Stage 2 | "Find disagreements between different buyers, not same person" |
| Unsupported insights | Evidence audit finds orphans | critical | Strip insight, re-run Stage 2 | "Remove insight '{text}' — no evidence" |
| Low truth density | <1.5 | critical | Re-run Query Gen → VoC (more evidence) | "Find 2nd source for single-source insights" |
| Invented content | Insight text not in any quote | critical | HARD STRIP, log violation | "Insight flagged as hallucination" |
| Tool failure | Try/catch | varies | Retry with backoff, alternate tool | "Switch from Reddit to G2 for this category" |
| Max loops exceeded | `loop_count > 3` | escalated | Output partial + gap report | "Manual research needed for {gaps}" |
| Duplicate queries | Query in executed_queries | minor | Skip and generate new | "Generate alternative query" |

### Escalation Protocol (v2)

```python
class EscalationReport(BaseModel):
    status: Literal["ESCALATED"]
    partial_packet: ResearchPacket      # what we have so far
    missing_items: list[str]            # specific gaps
    attempted_queries: list[str]        # everything we tried
    truth_density: float                # final density score
    evidence_coverage: float            # % insights supported
    validation_failures: list[ValidationFailure]  # detailed failures
    recommendation: str                 # human-readable next steps
```

### System-Level Guardrails (v2)

```python
GUARDRAILS = {
    # Content quality
    "no_generic_summaries": True,
    "no_fake_quotes": True,
    "no_invented_insights": True,
    "every_insight_needs_evidence": True,
    
    # Process integrity
    "no_skipping_nodes": True,
    "no_combining_roles": True,
    "no_proceeding_without_validation": True,
    "synthesis_must_be_two_stage": True,
    
    # Thresholds
    "min_truth_density": 1.5,
    "min_pain_quotes": 5,
    "min_failed_attempts": 3,
    "min_objections": 3,
    "min_switching_stories": 3,
    "min_comparisons": 3,
    "min_desires": 2,
    "min_contradictions": 2,
    "min_decision_moments": 3,
    "min_structured_failed_attempts": 5,
    "min_competitors": 2,
    
    # Loop limits
    "max_loops": 3,
    "max_queries_per_category": 8,
}
```

---

## 9. Output: Approved Research Packet Structure (v2)

The system MUST produce this exact structure (or escalate):

```yaml
approved_research_packet:
  
  # --- Identity ---
  icp:
    role: "SDR / Sales Development Representative"
    company_size: "100-500 employees"
    industry: "B2B SaaS"
    pain_summary: "Frustrated with complex sales engagement tools 
                   that require dedicated admin and training"
  
  # --- Raw Evidence (indexed by quote_id) ---
  raw_quotes:                                    # minimum 19
    - quote_id: "q-3f8a1bc2"
      text: "exact buyer quote"
      source: "r/sales"
      source_url: "https://..."
      category: "pain"
    # ... all quotes with unique IDs
  
  # --- Structured Findings (all linked to quote_ids) ---
  failed_attempts:                               # minimum 5
    - what_they_tried: "Hired a consultant to set up Outreach"
      why_it_failed: "Consultant left and nobody knew how to modify sequences"
      supporting_quote_ids: ["q-5f7g9bh6", "q-9d1e3gj7"]
  
  objections:                                    # minimum 5
    - insight_id: "obj-001"
      text: "Too expensive for a team under 10 SDRs"
      supporting_quote_ids: ["q-2d5e7cf3", "q-6a8b0de4"]
  
  decision_moments:                              # minimum 3
    - trigger: "Pricing jump at renewal from $75/user to $125/user"
      decision: "Switched to Apollo.io"
      supporting_quote_ids: ["q-4e6f8ag4"]
      alternatives_considered: ["Salesloft", "Mailshake"]
  
  contradictions:                                # minimum 2
    - quote_id_a: "q-3f8a1bc2"
      quote_id_b: "q-7c9d1eh6"
      statement_a_text: "Setup takes forever"
      statement_b_text: "Once set up, it's the best tool we've used"
      tension: "Complexity tolerable IF onboarding is supported"
      signal_strength: "strong"
  
  # --- Language ---
  category_language:
    - "sales engagement"
    - "sequences"
    - "cadences"
  
  voice_fingerprint:
    buyer_language: ["follow-up system", "outreach tool"]
    fake_language: ["revenue intelligence platform"]
    emotional_triggers: ["I just want it to work"]
    jargon_map:
      "drive user adoption": "get my team to actually use it"
      "intuitive search": "find anything without clicking 10 times"
  
  # --- Evidence Traceability (NEW) ---
  evidence_map:
    links:
      - insight_id: "pain-001"
        insight_text: "SDR teams abandon tools that require admin overhead"
        insight_type: "pain"
        supporting_quote_ids: ["q-3f8a1bc2", "q-9a2c4df1", "q-7b3e5af0"]
        confidence: "cross_validated"   # 3 different sources agree
      # ... all insights with evidence links
    
    truth_density: 2.89                  # evidence_units / insights
    unsupported_insights: 0
  
  proof_classification:
    social_proof: ["q-abc...", "q-def..."]
    data_proof: ["47 upvotes on complaint post", "$125/user pricing"]
    authority_proof: []
    experiential_proof: ["q-4e6f8ag4"]   # switching story = lived experience
  
  # --- Quality Metrics (NEW) ---
  truth_density: 2.89
  evidence_coverage: 1.0                 # 100% of insights have evidence
  total_quotes: 26
  total_insights: 18
  single_source_warnings: 3              # insights with only 1 source
```

---

## 10. Skills Audit (v2)

### Currently Available Skills (can use NOW)

| Skill | Relevance | How It Maps to Nodes |
|-------|-----------|---------------------|
| **reddit-insights** | 🔴 Critical | VoC Mining Node — semantic search for buyer language, pain points, switching stories |
| **voice-extractor** | 🔴 Critical | Voice Fingerprint Node — extracts tone patterns, signature phrases, anti-patterns |
| **competitive-ads-extractor** | 🟡 High | Competitor Analysis Node — pulls competitor ad messaging and positioning |
| **content-research-writer** | 🟡 High | Synthesis Stage 2 — research structuring, citation management |
| **lead-research-assistant** | 🟢 Medium | Product Understanding Node — ICP definition, company/product analysis |
| **master-copywriter** | 🟢 Medium | Downstream consumer — uses research packet to generate content |
| **email-marketing-bible** | 🟢 Medium | Downstream consumer — informs email-specific research requirements |

### Skills Needed But NOT Available (must build or source)

| Skill | Priority | Maps To | Why It's Needed |
|-------|----------|---------|----------------|
| **Query Generation Engine** | 🔴 Critical | Query Generation Layer | Generate diverse, typed, high-signal queries across categories. Must prevent duplicates |
| **G2/Review Scraper** | 🔴 Critical | VoC Mining | Structured review extraction from G2, Capterra, TrustRadius with role/company filters |
| **Evidence Mapper** | 🔴 Critical | Synthesis Stage 2 | Build and maintain the evidence_map linking insights to quote_ids |
| **Truth Density Calculator** | 🔴 Critical | Validation Node | Compute and enforce truth_density metric |
| **Contradiction Detector** | 🟡 High | Synthesis Stage 2 | Find conflicting statements, assess signal strength |
| **Web Content Extractor** | 🟡 High | Product + Competitor | Clean text extraction from arbitrary URLs |
| **Forum/Community Scraper** | 🟢 Medium | VoC Mining | HackerNews, Indie Hackers, Stack Overflow, niche forums |
| **Decision Journey Mapper** | 🟢 Medium | Synthesis Stage 2 | Extract triggers, decisions, alternatives from switching stories |

### Skill Architecture (v2)

```
research-agent/
├── skills/
│   ├── query-generator/         # NEW: Query Generation Layer
│   │   ├── SKILL.md
│   │   ├── query_templates.py   # pain/complaint/switching/comparison patterns
│   │   ├── dedup_engine.py      # prevents duplicate query patterns
│   │   └── source_diversifier.py # ensures multi-source coverage
│   │
│   ├── voc-miner/               # VoC Mining — upgraded for per-category
│   │   ├── SKILL.md
│   │   ├── reddit_queries.py
│   │   ├── review_scraper.py
│   │   └── category_enforcer.py # checks per-category minimums
│   │
│   ├── evidence-mapper/         # NEW: Evidence Mapping
│   │   ├── SKILL.md
│   │   ├── link_builder.py      # creates EvidenceLink objects
│   │   ├── density_calculator.py # truth_density computation
│   │   └── audit.py             # validates all references
│   │
│   ├── competitor-mapper/
│   │   ├── SKILL.md
│   │   ├── ad_extractor.py
│   │   └── positioning_analyzer.py
│   │
│   ├── voice-analyzer/
│   │   ├── SKILL.md
│   │   ├── language_classifier.py
│   │   └── emotional_triggers.py
│   │
│   ├── research-validator/      # UPGRADED with truth density
│   │   ├── SKILL.md
│   │   ├── category_coverage.py
│   │   ├── evidence_audit.py
│   │   ├── truth_density.py
│   │   └── retry_instructor.py  # generates per-failure retry instructions
│   │
│   └── synthesis-engine/        # SPLIT into two stages
│       ├── SKILL.md
│       ├── stage1_extractor.py  # raw extraction only
│       ├── stage2_patterns.py   # contradiction + pattern detection
│       └── contradiction_detector.py
```

---

## 11. Technology Stack & Dependencies

```
# Core
langchain >= 0.3.0
langgraph >= 0.2.0
pydantic >= 2.0

# LLM
openai >= 1.0                # or google-generativeai for Gemini
langchain-openai             # LLM provider

# Tools
tavily-python                # web search
praw                         # Reddit API (backup for reddit-insights)
beautifulsoup4               # HTML parsing
httpx                        # async HTTP requests
firecrawl-py                 # web content extraction (optional)

# Data
pandas                       # data manipulation for synthesis
tiktoken                     # token counting for context management

# Observability
langsmith                    # tracing & debugging (optional but recommended)
```

---

## 12. Design Decisions (Resolved)

| Question | Decision | Rationale |
|----------|----------|-----------|
| LLM per node | Different models per node | Flash for VoC mining (volume), Pro for Synthesis + Validation (reasoning quality) |
| Parallel execution | VoC + Product + Competitor run concurrently | They don't depend on each other; Product/Competitor use their own tools, not VoC data |
| Data persistence | Checkpoint to memory (MemorySaver) | Required for HITL interrupt/resume. Upgrade to SQLite/Postgres in production |
| HITL | Two optional checkpoints, toggle via `hitl_mode` | Post-VoC and Post-Synthesis. Skippable for autonomous runs |
| Rate limiting | Sequential tool execution within VoC node | Parallel across nodes is fine, but within a single node, tools run one at a time to respect API limits |
| Cost management | Estimated $3-10 per run | Flash for collection (~60% of tokens), Pro for synthesis/validation (~40% of tokens) |

