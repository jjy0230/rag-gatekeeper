"""Stage 2: Lightweight Reranking and Noise Chunk Filtering."""

import asyncio
import time
import logging
from typing import Optional
from rag_gatekeeper.config import settings
from rag_gatekeeper.engine.interface import DecisionEngine
from rag_gatekeeper.models import ChunkItem, ScoredChunk, RerankRequest, RerankResponse

logger = logging.getLogger(__name__)


class ChunkReranker:
    """Stage 2 Gatekeeper: Parallel relevance scoring and noise filtering.
    
    Scores each candidate chunk on a 1-5 scale and drops slices scoring below threshold.
    Includes anti-recall-cliff fallback and pre-truncation against long prefill delays.
    """

    def __init__(self, engine: DecisionEngine):
        self.engine = engine

    def _truncate_chunk(self, text: str, max_chars: int) -> str:
        """Pre-truncate chunk text to guard against long prefill attention dilution."""
        if len(text) <= max_chars:
            return text
        # Truncate and append ellipsis
        return text[:max_chars] + "..."

    async def _score_single_chunk(
        self,
        query: str,
        chunk: ChunkItem,
        semaphore: asyncio.Semaphore,
        threshold: float,
    ) -> ScoredChunk:
        """Score a single chunk under concurrency control."""
        async with semaphore:
            start_time = time.perf_counter()
            safe_text = self._truncate_chunk(chunk.content, settings.RERANK_MAX_CHUNK_CHARS)

            # Structure context so Query is shared as prefix across all chunks
            # Maximizes vLLM KV prefix cache hit rate
            context = f"Query: {query}\nCandidate Chunk: {safe_text}"
            system_instruction = (
                "Rate the relevance and information sufficiency of Candidate Chunk for answering Query. "
                "Score on 1.0 (completely irrelevant/noise) to 5.0 (directly answers query)."
            )

            score_res = await self.engine.judge_score(
                context=context,
                question="relevance_to_query",
                scale_min=1.0,
                scale_max=5.0,
                system_instruction=system_instruction,
            )

            elapsed = (time.perf_counter() - start_time) * 1000.0
            kept = score_res.score >= threshold

            return ScoredChunk(
                chunk=chunk,
                score=score_res.score,
                kept=kept,
                latency_ms=elapsed,
            )

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        start_time = time.perf_counter()
        threshold = request.score_threshold or settings.RERANK_SCORE_THRESHOLD
        semaphore = asyncio.Semaphore(settings.RERANK_CONCURRENCY)

        if not request.chunks:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return RerankResponse(
                total_evaluated=0,
                kept_count=0,
                filtered_chunks=[],
                fallback_applied=False,
                total_latency_ms=elapsed,
            )

        # Fire concurrent evaluation tasks for all candidate chunks
        tasks = [
            self._score_single_chunk(request.query, chunk, semaphore, threshold)
            for chunk in request.chunks
        ]
        scored_chunks: list[ScoredChunk] = await asyncio.gather(*tasks)

        # Filter chunks that passed threshold
        passed = [sc for sc in scored_chunks if sc.kept]
        # Sort by score descending
        passed.sort(key=lambda x: x.score, reverse=True)

        fallback_applied = False
        # Anti-Recall-Cliff Safeguard:
        # If every chunk scored below threshold, do not starve the downstream LLM.
        # Fallback to retaining top N scored chunks if configured.
        if not passed and settings.RERANK_FALLBACK_TOP_K > 0 and scored_chunks:
            sorted_all = sorted(scored_chunks, key=lambda x: x.score, reverse=True)
            fallback_chunks = sorted_all[: settings.RERANK_FALLBACK_TOP_K]
            for fc in fallback_chunks:
                fc.kept = True
            passed = fallback_chunks
            fallback_applied = True
            logger.warning(
                "All %d chunks scored below threshold %.2f; applied anti-cliff fallback (top %d retained).",
                len(scored_chunks),
                threshold,
                len(passed),
            )

        elapsed = (time.perf_counter() - start_time) * 1000.0

        return RerankResponse(
            total_evaluated=len(scored_chunks),
            kept_count=len(passed),
            filtered_chunks=passed,
            fallback_applied=fallback_applied,
            total_latency_ms=elapsed,
        )
