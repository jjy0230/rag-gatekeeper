"""FastAPI Gateway Server for RAG-Gatekeeper with UI Dashboard, Native Vector Store, and Audit Sink."""

import time
import logging
from pathlib import Path
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_gatekeeper.config import settings, reload_settings
from rag_gatekeeper.engine import get_engine
from rag_gatekeeper.stages.stage1_arbitration import RetrievalArbitrator
from rag_gatekeeper.stages.stage2_rerank import ChunkReranker
from rag_gatekeeper.stages.stage3_guardrail import HallucinationGuardrail
from rag_gatekeeper.pipeline import RAGGatekeeperPipeline
from rag_gatekeeper.proxy import GatewayProxy
from rag_gatekeeper.vector_store import InMemoryHybridStore
from rag_gatekeeper.audit_sink import audit_sink
from rag_gatekeeper.semantic_cache import semantic_cache
from rag_gatekeeper.models import (
    ArbitrationRequest,
    ArbitrationResponse,
    RerankRequest,
    RerankResponse,
    GuardrailRequest,
    GuardrailResponse,
    PipelineRequest,
    PipelineResponse,
    ChunkItem,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_gatekeeper")

app = FastAPI(
    title="RAG-Gatekeeper Enterprise",
    description="Edge dynamic short-circuiting, reranking & hallucination guardrail gateway powered by Jev",
    version="0.3.0",
)

# CORS middleware for Web/UI dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = get_engine()
stage1 = RetrievalArbitrator(engine)
stage2 = ChunkReranker(engine)
stage3 = HallucinationGuardrail(engine)
pipeline = RAGGatekeeperPipeline(engine)
gateway_proxy = GatewayProxy(stage1, stage2, stage3)
vector_store = InMemoryHybridStore()

# Global Telemetry Counters
metrics_data = {
    "requests_total": 0,
    "stage1_short_circuited_total": 0,
    "stage1_retrieval_required_total": 0,
    "stage2_chunks_evaluated_total": 0,
    "stage2_chunks_kept_total": 0,
    "stage3_blocked_total": 0,
}


@app.get("/healthz")
async def health_check():
    """Liveness & readiness probe."""
    return {
        "status": "healthy",
        "mock_engine": settings.USE_MOCK_ENGINE,
        "jev_endpoint": settings.JEV_ENDPOINT,
        "upstream_configured": bool(settings.UPSTREAM_LLM_URL),
        "version": "0.3.0-enterprise",
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve enterprise interactive management web console."""
    html_path = Path(__file__).resolve().parent / "ui" / "dashboard.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard UI file not found</h1>", status_code=404)


@app.get("/metrics")
async def prometheus_metrics():
    """Expose Prometheus-compatible plaintext metrics."""
    lines = [
        "# HELP rag_gk_requests_total Total number of processed requests.",
        "# TYPE rag_gk_requests_total counter",
        f"rag_gk_requests_total {metrics_data['requests_total']}",
        "# HELP rag_gk_short_circuited_total Requests short-circuited at Stage 1.",
        "# TYPE rag_gk_short_circuited_total counter",
        f"rag_gk_short_circuited_total {metrics_data['stage1_short_circuited_total']}",
        "# HELP rag_gk_retrieval_required_total Requests routed to vector retrieval.",
        "# TYPE rag_gk_retrieval_required_total counter",
        f"rag_gk_retrieval_required_total {metrics_data['stage1_retrieval_required_total']}",
        "# HELP rag_gk_chunks_evaluated_total Total chunks scored by Jev.",
        "# TYPE rag_gk_chunks_evaluated_total counter",
        f"rag_gk_chunks_evaluated_total {metrics_data['stage2_chunks_evaluated_total']}",
        "# HELP rag_gk_chunks_kept_total Chunks passing Stage 2 filter.",
        "# TYPE rag_gk_chunks_kept_total counter",
        f"rag_gk_chunks_kept_total {metrics_data['stage2_chunks_kept_total']}",
        "# HELP rag_gk_blocked_total Answers blocked by Stage 3 guardrail.",
        "# TYPE rag_gk_blocked_total counter",
        f"rag_gk_blocked_total {metrics_data['stage3_blocked_total']}",
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


class ConfigReloadRequest(BaseModel):
    FORCE_RETRIEVAL_KEYWORDS: Optional[list[str]] = None
    RERANK_SCORE_THRESHOLD: Optional[float] = None
    HALLUCINATION_PROB_THRESHOLD: Optional[float] = None
    RERANK_FALLBACK_TOP_K: Optional[int] = None
    UPSTREAM_LLM_URL: Optional[str] = None


@app.post("/v1/gatekeeper/config/reload")
async def reload_configuration(req: ConfigReloadRequest):
    """Hot-reload gateway parameters and Aho-Corasick white-list without restart."""
    updates = req.model_dump(exclude_unset=True)
    reload_settings(updates)
    if req.FORCE_RETRIEVAL_KEYWORDS is not None:
        stage1.reload_keywords(req.FORCE_RETRIEVAL_KEYWORDS)
    logger.info("Gateway configuration hot-reloaded: %s", updates)
    return {"status": "reloaded", "applied_updates": updates}


# ==========================================
# Semantic Cache Endpoints
# ==========================================

@app.get("/v1/gatekeeper/cache/stats")
async def get_cache_statistics():
    """Retrieve semantic cache hit/miss statistics and entry counts."""
    return semantic_cache.stats()


@app.post("/v1/gatekeeper/cache/clear")
async def clear_cache():
    """Clear all semantic cache entries."""
    semantic_cache.clear()
    return {"status": "cleared"}


# ==========================================
# Native Vector Store & Knowledge Endpoints
# ==========================================

class UpsertKnowledgeRequest(BaseModel):
    chunks: list[ChunkItem]


@app.post("/v1/gatekeeper/knowledge/upsert")
async def upsert_knowledge_base(req: UpsertKnowledgeRequest):
    """Upsert documents into native in-memory hybrid vector store."""
    vector_store.upsert_chunks(req.chunks)
    return {"status": "success", "count": len(req.chunks)}


class SearchKnowledgeRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/v1/gatekeeper/knowledge/search", response_model=list[ChunkItem])
async def search_knowledge_base(req: SearchKnowledgeRequest):
    """Search native hybrid vector store."""
    return vector_store.search(req.query, top_k=req.top_k)


# ==========================================
# Audit & Training Dataset Export
# ==========================================

@app.post("/v1/gatekeeper/audit/export")
async def export_audit_training_pairs():
    """Export Jev calibration pairs from local audit sink."""
    count = audit_sink.export_jev_finetuning_dataset()
    return {"status": "exported", "pairs_count": count, "output_path": "data/jev_training_pairs.jsonl"}


# ==========================================
# Core Decision Gatekeeper Endpoints
# ==========================================

@app.post("/v1/gatekeeper/arbitrate", response_model=ArbitrationResponse)
async def arbitrate_retrieval(req: ArbitrationRequest):
    """Stage 1: Decide whether query needs external knowledge retrieval."""
    metrics_data["requests_total"] += 1
    try:
        res = await stage1.arbitrate(req)
        if res.needs_external_knowledge:
            metrics_data["stage1_retrieval_required_total"] += 1
        else:
            metrics_data["stage1_short_circuited_total"] += 1

        # Audit logging
        await audit_sink.log_event("STAGE1_ARBITRATION", {
            "query": req.query,
            "needs_external_knowledge": res.needs_external_knowledge,
            "confidence": res.confidence,
        })
        return res
    except Exception as exc:
        logger.error("Arbitration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/gatekeeper/rerank", response_model=RerankResponse)
async def rerank_chunks(req: RerankRequest):
    """Stage 2: Score chunks on 1-5 scale and filter low-scoring slices."""
    try:
        res = await stage2.rerank(req)
        metrics_data["stage2_chunks_evaluated_total"] += res.total_evaluated
        metrics_data["stage2_chunks_kept_total"] += res.kept_count
        return res
    except Exception as exc:
        logger.error("Rerank error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/gatekeeper/verify", response_model=GuardrailResponse)
async def verify_hallucination(req: GuardrailRequest):
    """Stage 3: Verify factual consistency and block hallucinated responses."""
    try:
        res = await stage3.verify(req)
        if res.blocked:
            metrics_data["stage3_blocked_total"] += 1
            await audit_sink.log_event("STAGE3_HALLUCINATION_BLOCKED", {
                "query": req.query,
                "answer": req.answer,
                "evidence": req.evidence_chunks,
                "hallucination_prob": res.hallucination_prob,
            })
        return res
    except Exception as exc:
        logger.error("Guardrail error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/gatekeeper/pipeline", response_model=PipelineResponse)
async def run_pipeline(req: PipelineRequest):
    """Execute end-to-end gatekeeper arbitration, rerank & verification."""
    metrics_data["requests_total"] += 1
    try:
        # If no candidate chunks provided, automatically retrieve from native hybrid store
        chunks_to_use = req.candidate_chunks
        if not chunks_to_use and vector_store._chunks:
            chunks_to_use = vector_store.search(req.query, top_k=5)

        request_with_chunks = req.model_copy(update={"candidate_chunks": chunks_to_use})
        res = await pipeline.execute(request_with_chunks)

        if res.short_circuited:
            metrics_data["stage1_short_circuited_total"] += 1
        else:
            metrics_data["stage1_retrieval_required_total"] += 1
        return res
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ==========================================
# OpenAI-Compatible Reverse Proxy Endpoint
# ==========================================

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """OpenAI-compatible reverse proxy endpoint supporting streaming (SSE) and non-streaming."""
    metrics_data["requests_total"] += 1
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    return await gateway_proxy.handle_chat_completion(body, request)
