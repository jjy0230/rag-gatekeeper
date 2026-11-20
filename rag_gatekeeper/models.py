"""Data models and schemas for RAG-Gatekeeper."""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ==========================================
# Jev Decision Primitive Contracts
# ==========================================

class NoulResult(BaseModel):
    """Jev Noul decision primitive (calibrated Boolean judgment + probability)."""
    value: bool = Field(..., description="Boolean judgment (True/False)")
    prob_true: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability that the statement is True")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of the decision (max(p, 1-p))")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    raw: Optional[dict[str, Any]] = None


class ScoreResult(BaseModel):
    """Jev Score decision primitive (rating on a continuous or discrete scale)."""
    score: float = Field(..., ge=1.0, le=5.0, description="Rating score [1.0 - 5.0]")
    distribution: Optional[dict[int, float]] = Field(
        default=None, description="Probability distribution across 1-5"
    )
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    raw: Optional[dict[str, Any]] = None


class ChoiceResult(BaseModel):
    """Jev Choice decision primitive (categorical classification)."""
    selected: str = Field(..., description="Selected option name")
    probabilities: dict[str, float] = Field(
        default_factory=dict, description="Probability across choices"
    )
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    raw: Optional[dict[str, Any]] = None


# ==========================================
# Stage 1: Retrieval Arbitration Schemas
# ==========================================

class ArbitrationRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or search query")
    knowledge_base_desc: Optional[str] = Field(
        default=None, description="Optional brief description of the target knowledge base"
    )


class ArbitrationResponse(BaseModel):
    needs_external_knowledge: bool = Field(
        ..., description="Whether vector retrieval is required"
    )
    confidence: float = Field(..., description="Confidence of the decision")
    latency_ms: float = Field(..., description="Time spent on arbitration")
    reasoning_mode: str = Field(
        default="jev_noul", description="Mode: jev_noul | keyword_bypass | fallback"
    )


# ==========================================
# Stage 2: Rerank & Filtering Schemas
# ==========================================

class ChunkItem(BaseModel):
    id: str = Field(..., description="Unique identifier of the document chunk")
    content: str = Field(..., description="Text content of the retrieved chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata associated with chunk")


class ScoredChunk(BaseModel):
    chunk: ChunkItem
    score: float = Field(..., ge=1.0, le=5.0, description="Relevance score evaluated by Jev")
    kept: bool = Field(..., description="Whether this chunk passed the threshold")
    latency_ms: float = Field(..., description="Evaluation latency for this chunk")


class RerankRequest(BaseModel):
    query: str = Field(..., description="User query")
    chunks: list[ChunkItem] = Field(..., description="Candidate chunks recalled by vector DB")
    score_threshold: Optional[float] = Field(
        default=None, description="Override score cutoff (defaults to config: 4.0)"
    )


class RerankResponse(BaseModel):
    total_evaluated: int
    kept_count: int
    filtered_chunks: list[ScoredChunk]
    fallback_applied: bool = Field(
        default=False, description="True if anti-recall-cliff fallback was triggered"
    )
    total_latency_ms: float


# ==========================================
# Stage 3: Hallucination Guardrail Schemas
# ==========================================

class GuardrailRequest(BaseModel):
    query: str = Field(..., description="User query")
    answer: str = Field(..., description="Generated answer from LLM")
    evidence_chunks: list[str] = Field(
        ..., description="Text of filtered chunks provided as reference context"
    )


class GuardrailResponse(BaseModel):
    blocked: bool = Field(..., description="True if answer is blocked due to hallucination")
    hallucination_prob: float = Field(..., description="Probability of containing unsupported facts")
    final_answer: str = Field(..., description="Safe final answer (either original or degraded template)")
    latency_ms: float
    degraded: bool = Field(..., description="True if response was replaced by fallback template")


# ==========================================
# Full Gateway Pipeline Schemas
# ==========================================

class PipelineRequest(BaseModel):
    query: str = Field(..., description="User query")
    candidate_chunks: Optional[list[ChunkItem]] = Field(
        default=None, description="Optional pre-retrieved candidate chunks from Vector DB"
    )
    generated_answer: Optional[str] = Field(
        default=None, description="Optional pre-generated LLM answer (if testing Stage 3 directly)"
    )
    last_turn_query: Optional[str] = Field(
        default=None, description="Optional previous query for multi-turn pronoun de-referencing"
    )
    enable_semantic_cache: bool = Field(default=True, description="Whether to check semantic cache first")
    enable_citations: bool = Field(default=True, description="Whether to inject [1],[2] citation footnotes")
    enable_compression: bool = Field(default=True, description="Whether to apply sentence-level context compression")


class PipelineResponse(BaseModel):
    query: str
    short_circuited: bool = Field(
        ..., description="True if vector retrieval was bypassed in Stage 1"
    )
    cache_hit: bool = Field(default=False, description="True if answer was served instantly from semantic cache")
    arbitration: Optional[ArbitrationResponse] = None
    rerank: Optional[RerankResponse] = None
    guardrail: Optional[GuardrailResponse] = None
    final_output: Optional[str] = None
    citations: list[dict] = Field(default_factory=list, description="Grounding source citations")
    total_overhead_latency_ms: float
    rewritten_query: Optional[str] = None
