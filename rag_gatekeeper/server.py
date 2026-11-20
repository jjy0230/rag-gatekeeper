"""FastAPI Gateway Server for RAG-Gatekeeper."""

import time
import logging
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_gatekeeper.config import settings
from rag_gatekeeper.engine import get_engine
from rag_gatekeeper.stages.stage1_arbitration import RetrievalArbitrator
from rag_gatekeeper.stages.stage2_rerank import ChunkReranker
from rag_gatekeeper.stages.stage3_guardrail import HallucinationGuardrail
from rag_gatekeeper.pipeline import RAGGatekeeperPipeline
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
    version="0.1.0",
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


@app.get("/healthz")
async def health_check():
    """Liveness probe."""
    return {
        "status": "healthy",
        "mock_engine": settings.USE_MOCK_ENGINE,
        "jev_endpoint": settings.JEV_ENDPOINT,
    }


@app.post("/v1/gatekeeper/arbitrate", response_model=ArbitrationResponse)
async def arbitrate_retrieval(req: ArbitrationRequest):
    """Stage 1: Decide whether query needs external knowledge retrieval."""
    try:
        return await stage1.arbitrate(req)
    except Exception as exc:
        logger.error("Arbitration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/gatekeeper/rerank", response_model=RerankResponse)
async def rerank_chunks(req: RerankRequest):
    """Stage 2: Score chunks on 1-5 scale and filter low-scoring slices."""
    try:
        return await stage2.rerank(req)
    except Exception as exc:
        logger.error("Rerank error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/gatekeeper/verify", response_model=GuardrailResponse)
async def verify_hallucination(req: GuardrailRequest):
    """Stage 3: Verify factual consistency and block hallucinated responses."""
    try:
        return await stage3.verify(req)
    except Exception as exc:
        logger.error("Guardrail error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/gatekeeper/pipeline", response_model=PipelineResponse)
async def run_pipeline(req: PipelineRequest):
    """Execute end-to-end gatekeeper arbitration, rerank & verification."""
    try:
        return await pipeline.execute(req)
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ==========================================
# OpenAI-Compatible Reverse Proxy Endpoint
# (Allows zero-code integration with Dify, LangChain, LlamaIndex)
# ==========================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7


@app.post("/v1/chat/completions")
async def proxy_chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible reverse proxy endpoint.
    
    Intercepts user query, evaluates Stage 1 arbitration gate, and returns
    gateway telemetry in the standard chat completions response.
    """
    user_query = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            user_query = msg.content
            break

    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in request")

    # Run Stage 1 Arbitration
    arb_res = await stage1.arbitrate(ArbitrationRequest(query=user_query))

    response_content = (
        f"【Gatekeeper 路由】\n"
        f"- 检索短路状态: {'[已短路-跳过检索]' if not arb_res.needs_external_knowledge else '[放行-需检索向量库]'}\n"
        f"- 决策延迟: {arb_res.latency_ms:.2f}ms\n"
        f"- 置信度: {arb_res.confidence:.2f}\n\n"
        f"（当前网关运行在接口模拟模式，下游大模型转发已就绪）"
    )

    return {
        "id": f"chatcmpl-gk-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "gatekeeper_telemetry": {
            "stage1_arbitration": arb_res.model_dump(),
        },
    }
