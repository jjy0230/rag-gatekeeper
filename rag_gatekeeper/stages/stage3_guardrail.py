"""Stage 3: Hallucination and Factuality Gatekeeper."""

import time
import logging
from rag_gatekeeper.config import settings
from rag_gatekeeper.engine.interface import DecisionEngine
from rag_gatekeeper.models import GuardrailRequest, GuardrailResponse

logger = logging.getLogger(__name__)


class HallucinationGuardrail:
    """Stage 3 Gatekeeper: Output verification & Hallucination blocking.
    
    Verifies whether the generated answer contains ungrounded or contradictory claims
    against the retrieved evidence chunks. Compresses traditional LLM-as-a-judge
    verification (2~4s) down to ~25ms.
    """

    def __init__(self, engine: DecisionEngine):
        self.engine = engine

    async def verify(self, request: GuardrailRequest) -> GuardrailResponse:
        start_time = time.perf_counter()

        # Format context: evidence chunks + generated answer
        formatted_evidence = "\n---\n".join(
            [f"[Evidence {i+1}]: {chunk}" for i, chunk in enumerate(request.evidence_chunks)]
        )
        context = (
            f"Grounding Evidence:\n{formatted_evidence}\n\n"
            f"User Query:\n{request.query}\n\n"
            f"Generated Answer:\n{request.answer}"
        )

        system_instruction = (
            "Verify factual consistency. Determine whether the Generated Answer contains "
            "unsupported claims, hallucinated facts, or statements contradicting the Grounding Evidence. "
            "Return True if unsupported facts are present, False if fully supported."
        )

        noul_res = await self.engine.judge_noul(
            context=context,
            question="contains_unsupported_facts",
            system_instruction=system_instruction,
        )

        # In Jev, prob_true is the calibrated posterior probability that the statement is True.
        # Here, prob_true directly represents the probability that answer contains unsupported facts.
        hallucination_prob = noul_res.prob_true

        blocked = hallucination_prob > settings.HALLUCINATION_PROB_THRESHOLD
        elapsed = (time.perf_counter() - start_time) * 1000.0

        if blocked and settings.GUARDRAIL_MODE == "BLOCK_AND_DEGRADE":
            logger.warning(
                "Hallucination detected (prob=%.2f > threshold=%.2f). Blocking response.",
                hallucination_prob,
                settings.HALLUCINATION_PROB_THRESHOLD,
            )
            final_answer = settings.FALLBACK_TEMPLATE
            degraded = True
        else:
            final_answer = request.answer
            degraded = False

        return GuardrailResponse(
            blocked=blocked,
            hallucination_prob=hallucination_prob,
            final_answer=final_answer,
            latency_ms=elapsed,
            degraded=degraded,
        )
