"""Stage 1: Retrieval Arbitration Gatekeeper (Short-circuiting)."""

import time
import logging
from rag_gatekeeper.config import settings
from rag_gatekeeper.engine.interface import DecisionEngine
from rag_gatekeeper.models import ArbitrationRequest, ArbitrationResponse

logger = logging.getLogger(__name__)


class RetrievalArbitrator:
    """Stage 1 Gatekeeper: Arbitrates whether a query warrants external vector DB retrieval.
    
    Prevents trivial or conversational queries from unnecessarily executing expensive
    vector similarity searches, reducing vector database workload by 30%~40%.
    """

    def __init__(self, engine: DecisionEngine):
        self.engine = engine

    async def arbitrate(self, request: ArbitrationRequest) -> ArbitrationResponse:
        start_time = time.perf_counter()
        query_text = request.query.strip()

        # Rule-based fast bypass: check if query explicitly hits enterprise domain keywords
        if settings.FORCE_RETRIEVAL_KEYWORDS:
            for kw in settings.FORCE_RETRIEVAL_KEYWORDS:
                if kw in query_text:
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                    return ArbitrationResponse(
                        needs_external_knowledge=True,
                        confidence=1.0,
                        latency_ms=elapsed,
                        reasoning_mode="keyword_bypass",
                    )

        # Context construction for Jev
        context = f"User Query: {query_text}"
        if request.knowledge_base_desc:
            context = f"Knowledge Base Scope: {request.knowledge_base_desc}\n{context}"

        system_instruction = (
            "Decide whether the user query requires external knowledge base retrieval. "
            "Return False for general greetings, general programming, or common knowledge. "
            "Return True for specific facts, domain knowledge, policies, or private data."
        )

        noul_result = await self.engine.judge_noul(
            context=context,
            question="needs_external_knowledge",
            system_instruction=system_instruction,
        )

        elapsed = (time.perf_counter() - start_time) * 1000.0

        return ArbitrationResponse(
            needs_external_knowledge=noul_result.value,
            confidence=noul_result.confidence,
            latency_ms=elapsed,
            reasoning_mode="jev_noul",
        )
