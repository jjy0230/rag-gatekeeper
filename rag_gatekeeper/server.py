"""FastAPI Gateway Server for RAG-Gatekeeper with streaming proxy and dynamic policy reload."""

import time
import logging
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_gatekeeper.config import settings, reload_settings
from rag_gatekeeper.engine import get_engine
from rag_gatekeeper.stages.stage1_arbitration import RetrievalArbitrator
from rag_gatekeeper.stages.stage2_rerank import ChunkReranker
from rag_gatekeeper.stages.stage3_guardrail import HallucinationGuardrail
from rag_gatekeeper.pipeline import RAGGatekeeperPipeline
from rag_gatekeeper.proxy import GatewayProxy
from rag_gatekeeper.models import (
    ArbitrationRequest,
    ArbitrationResponse,
    RerankRequest,
    RerankResponse,
    GuardrailRequest,
    GuardrailResponse,
    PipelineRequest,
    PipelineResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_gatekeeper")

app = FastAPI(
    title="RAG-Gatekeeper",
    description="Edge dynamic short-circuiting, reranking & hallucination guardrail gateway powered by Jev",
    version="0.2.0",
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
        "version": "0.2.0",
    }


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
        return res
    except Exception as exc:
        logger.error("Guardrail error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/gatekeeper/pipeline", response_model=PipelineResponse)
async def run_pipeline(req: PipelineRequest):
    """Execute end-to-end gatekeeper arbitration, rerank & verification."""
    metrics_data["requests_total"] += 1
    try:
        res = await pipeline.execute(req)
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
# (Allows zero-code integration with Dify, LangChain, LlamaIndex)
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
