"""End-to-end RAG Gatekeeper Pipeline orchestration with Domain Routing, Semantic Cache, Compression, Citations & Confidence Radar."""

import time
import logging
from typing import Callable, Awaitable, Optional
from rag_gatekeeper.engine.interface import DecisionEngine
from rag_gatekeeper.stages.stage1_arbitration import RetrievalArbitrator
from rag_gatekeeper.stages.stage2_rerank import ChunkReranker
from rag_gatekeeper.stages.stage3_guardrail import HallucinationGuardrail
from rag_gatekeeper.domain_router import DomainRouter, DomainRoutingResult
from rag_gatekeeper.confidence_radar import confidence_radar
from rag_gatekeeper.semantic_cache import semantic_cache
from rag_gatekeeper.query_rewriter import query_rewriter
from rag_gatekeeper.context_compressor import context_compressor
from rag_gatekeeper.citation_injector import citation_injector
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
    """Orchestrates Semantic Cache -> De-referencing -> Stage 1 -> Domain Routing ->

    Retrieval -> Stage 2 -> Compression -> LLM -> Stage 3 -> Radar & Footnote Citations.
    """

    def __init__(self, engine: DecisionEngine):
        self.engine = engine
        self.stage1 = RetrievalArbitrator(engine)
        self.stage2 = ChunkReranker(engine)
        self.stage3 = HallucinationGuardrail(engine)
        self.domain_router = DomainRouter(engine)

    async def execute(
        self,
        request: PipelineRequest,
        retriever_hook: Optional[Callable[[str, Optional[str]], Awaitable[list[ChunkItem]]]] = None,
        generator_hook: Optional[Callable[[str, list[str]], Awaitable[str]]] = None,
    ) -> PipelineResponse:
        start_time = time.perf_counter()

        # ----------------------------------------------------
        # Step 0: Semantic Cache Gatekeeper (< 2ms)
        # ----------------------------------------------------
        if request.enable_semantic_cache:
            cached_entry = semantic_cache.get(request.query)
            if cached_entry:
                overhead = (time.perf_counter() - start_time) * 1000.0
                logger.info("Semantic cache HIT for query: %s", request.query)
                return PipelineResponse(
                    query=request.query,
                    short_circuited=True,
                    cache_hit=True,
                    arbitration=None,
                    rerank=None,
                    guardrail=None,
                    final_output=cached_entry["response"],
                    citations=cached_entry.get("citations", []),
                    total_overhead_latency_ms=overhead,
                    confidence_tier="AUTHORITATIVE_VERIFIED",
                )

        # ----------------------------------------------------
        # Step 0.5: Multi-turn Pronoun De-referencing & Query Rewriting
        # ----------------------------------------------------
        active_query, was_rewritten = query_rewriter.rewrite_query(
            request.query, request.last_turn_query
        )

        # ----------------------------------------------------
        # Stage 1: Retrieval Arbitration Gate (~8ms)
        # ----------------------------------------------------
        arb_res = await self.stage1.arbitrate(ArbitrationRequest(query=active_query))

        # Check if short-circuited
        if not arb_res.needs_external_knowledge:
            logger.info("Stage 1 Short-circuited: No retrieval needed for query: %s", active_query)
            final_output = None
            if generator_hook:
                final_output = await generator_hook(active_query, [])
            elif request.generated_answer:
                final_output = request.generated_answer
            else:
                final_output = "【直连回答】此问题判定为通用常识或无需外部知识库支持，已绕过向量数据库直接生成。"

            overhead = (time.perf_counter() - start_time) * 1000.0
            if request.enable_semantic_cache:
                semantic_cache.set(request.query, final_output)

            return PipelineResponse(
                query=request.query,
                short_circuited=True,
                cache_hit=False,
                arbitration=arb_res,
                rerank=None,
                guardrail=None,
                final_output=final_output,
                citations=[],
                total_overhead_latency_ms=overhead,
                rewritten_query=active_query if was_rewritten else None,
                confidence_tier="AUTHORITATIVE_VERIFIED",
            )

        # ----------------------------------------------------
        # Step 1.5: Multi-Domain Knowledge Routing (Jev Choice)
        # ----------------------------------------------------
        route_res = await self.domain_router.route_query(active_query)
        target_partition = route_res.target_partition_key
        logger.info("Query routed to domain partition: %s (%s)", route_res.target_domain, target_partition)

        # ----------------------------------------------------
        # Vector Retrieval Phase (with Partition Isolation)
        # ----------------------------------------------------
        candidate_chunks: list[ChunkItem] = []
        if retriever_hook:
            # Pass partition_key if retriever hook supports it
            try:
                candidate_chunks = await retriever_hook(active_query, target_partition)
            except TypeError:
                candidate_chunks = await retriever_hook(active_query)
        elif request.candidate_chunks:
            candidate_chunks = request.candidate_chunks

        # ----------------------------------------------------
        # Stage 2: Rerank & Noise Filtering (~25ms)
        # ----------------------------------------------------
        rerank_res = await self.stage2.rerank(
            RerankRequest(query=active_query, chunks=candidate_chunks)
        )

        kept_chunk_objects = [sc.chunk for sc in rerank_res.filtered_chunks]

        # ----------------------------------------------------
        # Step 2.5: High-density Extractive Sentence Compression
        # ----------------------------------------------------
        if request.enable_compression and kept_chunk_objects:
            compressed_chunks = context_compressor.compress_chunks(active_query, kept_chunk_objects)
            effective_chunks = compressed_chunks
        else:
            effective_chunks = kept_chunk_objects

        filtered_contexts = [c.content for c in effective_chunks]

        # ----------------------------------------------------
        # LLM Generation Phase
        # ----------------------------------------------------
        generated_answer = request.generated_answer
        if generator_hook and not generated_answer:
            generated_answer = await generator_hook(active_query, filtered_contexts)
        elif not generated_answer:
            generated_answer = (
                f"根据参考文档（共 {len(filtered_contexts)} 条），这是对于问题 '{active_query}' 的综合解答。"
            )

        # ----------------------------------------------------
        # Stage 3: Hallucination Guardrail (~25ms)
        # ----------------------------------------------------
        guardrail_res = await self.stage3.verify(
            GuardrailRequest(
                query=active_query,
                answer=generated_answer,
                evidence_chunks=filtered_contexts,
            )
        )

        # ----------------------------------------------------
        # Step 3.2: 3-Tier Confidence Radar & Adaptive Disclaimers
        # ----------------------------------------------------
        tier_res = confidence_radar.evaluate_tier(
            guardrail_res.final_answer,
            guardrail_res.hallucination_prob,
        )

        final_output_text = tier_res.final_output
        citation_records = []

        # ----------------------------------------------------
        # Step 3.5: Citation Footnote Injection
        # ----------------------------------------------------
        if request.enable_citations and tier_res.tier != "HALLUCINATED_BLOCKED" and effective_chunks:
            cite_res = citation_injector.inject_citations(final_output_text, effective_chunks)
            final_output_text = cite_res.annotated_answer
            citation_records = [c.model_dump() for c in cite_res.citations]

        overhead = (time.perf_counter() - start_time) * 1000.0

        if request.enable_semantic_cache and tier_res.tier != "HALLUCINATED_BLOCKED":
            semantic_cache.set(request.query, final_output_text, citations=citation_records)

        return PipelineResponse(
            query=request.query,
            short_circuited=False,
            cache_hit=False,
            arbitration=arb_res,
            rerank=rerank_res,
            guardrail=guardrail_res,
            final_output=final_output_text,
            citations=citation_records,
            total_overhead_latency_ms=overhead,
            rewritten_query=active_query if was_rewritten else None,
            routed_domain=route_res.target_domain,
            confidence_tier=tier_res.tier,
            disclaimer_applied=tier_res.disclaimer_added,
        )
