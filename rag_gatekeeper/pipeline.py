"""End-to-end RAG Gatekeeper Pipeline orchestration."""

import time
import logging
from typing import Callable, Awaitable, Optional
from rag_gatekeeper.engine.interface import DecisionEngine
from rag_gatekeeper.stages.stage1_arbitration import RetrievalArbitrator
from rag_gatekeeper.stages.stage2_rerank import ChunkReranker
from rag_gatekeeper.stages.stage3_guardrail import HallucinationGuardrail
from rag_gatekeeper.models import (
    ArbitrationRequest,
    RerankRequest,
    GuardrailRequest,
    PipelineRequest,
    PipelineResponse,
    ChunkItem,
)

logger = logging.getLogger(__name__)


class RAGGatekeeperPipeline:
    """Orchestrates Stage 1 (Arbitration), Vector Retrieval, Stage 2 (Rerank),

    LLM Generation, and Stage 3 (Guardrail) in a clean, observable pipeline.
    """

    def __init__(self, engine: DecisionEngine):
        self.engine = engine
        self.stage1 = RetrievalArbitrator(engine)
        self.stage2 = ChunkReranker(engine)
        self.stage3 = HallucinationGuardrail(engine)

    async def execute(
        self,
        request: PipelineRequest,
        retriever_hook: Optional[Callable[[str], Awaitable[list[ChunkItem]]]] = None,
        generator_hook: Optional[Callable[[str, list[str]], Awaitable[str]]] = None,
    ) -> PipelineResponse:
        """Run the end-to-end gatekeeper workflow.
        
        Args:
            request: Pipeline input data.
            retriever_hook: Optional async callable (query: str) -> list[ChunkItem].
            generator_hook: Optional async callable (query: str, contexts: list[str]) -> str.
            
        Returns:
            PipelineResponse containing execution details and intermediate stage metrics.
        """
        start_time = time.perf_counter()

        # ----------------------------------------------------
        # Stage 1: Retrieval Arbitration Gate (~8ms)
        # ----------------------------------------------------
        arb_res = await self.stage1.arbitrate(ArbitrationRequest(query=request.query))

        # Check if short-circuited
        if not arb_res.needs_external_knowledge:
            logger.info("Stage 1 Short-circuited: No retrieval needed for query: %s", request.query)
            # Directly invoke generator without retrieval if provided
            final_output = None
            if generator_hook:
                final_output = await generator_hook(request.query, [])
            elif request.generated_answer:
                final_output = request.generated_answer
            else:
                final_output = "【直连回答】此问题判定为通用常识或无需外部知识库支持，已绕过向量数据库直接生成。"

            overhead = (time.perf_counter() - start_time) * 1000.0
            return PipelineResponse(
                query=request.query,
                short_circuited=True,
                arbitration=arb_res,
                rerank=None,
                guardrail=None,
                final_output=final_output,
                total_overhead_latency_ms=overhead,
            )

        # ----------------------------------------------------
        # Vector Retrieval Phase
        # ----------------------------------------------------
        candidate_chunks: list[ChunkItem] = []
        if retriever_hook:
            candidate_chunks = await retriever_hook(request.query)
        elif request.candidate_chunks:
            candidate_chunks = request.candidate_chunks

        # ----------------------------------------------------
        # Stage 2: Rerank & Noise Filtering (~25ms)
        # ----------------------------------------------------
        rerank_res = await self.stage2.rerank(
            RerankRequest(query=request.query, chunks=candidate_chunks)
        )

        filtered_contexts = [sc.chunk.content for sc in rerank_res.filtered_chunks]

        # ----------------------------------------------------
        # LLM Generation Phase
        # ----------------------------------------------------
        generated_answer = request.generated_answer
        if generator_hook and not generated_answer:
            generated_answer = await generator_hook(request.query, filtered_contexts)
        elif not generated_answer:
            generated_answer = (
                f"根据参考文档（共 {len(filtered_contexts)} 条），这是对于问题 '{request.query}' 的综合解答。"
            )

        # ----------------------------------------------------
        # Stage 3: Hallucination Guardrail (~25ms)
        # ----------------------------------------------------
        guardrail_res = await self.stage3.verify(
            GuardrailRequest(
                query=request.query,
                answer=generated_answer,
                evidence_chunks=filtered_contexts,
            )
        )

        overhead = (time.perf_counter() - start_time) * 1000.0

        return PipelineResponse(
            query=request.query,
            short_circuited=False,
            arbitration=arb_res,
            rerank=rerank_res,
            guardrail=guardrail_res,
            final_output=guardrail_res.final_answer,
            total_overhead_latency_ms=overhead,
        )
