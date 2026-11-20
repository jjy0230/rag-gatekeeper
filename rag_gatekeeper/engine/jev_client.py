"""HTTP client for mode-io/vllm-jev serving engine."""

import time
import logging
from typing import Optional
import httpx
from rag_gatekeeper.config import settings
from rag_gatekeeper.engine.interface import DecisionEngine
from rag_gatekeeper.models import NoulResult, ScoreResult, ChoiceResult

logger = logging.getLogger(__name__)


class VllmJevClient(DecisionEngine):
    """Production client communicating with mode-io/vllm-jev service.
    
    Adheres to the Jev System One HTTP contract (POST /v1/systemone).
    Leverages persistent HTTP connection pooling and handles prefix caching alignments.
    """

    def __init__(
        self,
        endpoint: str = settings.JEV_ENDPOINT,
        model_name: str = settings.JEV_MODEL_NAME,
        timeout_ms: int = settings.JEV_TIMEOUT_MS,
    ):
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout = timeout_ms / 1000.0
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
            self._client = httpx.AsyncClient(timeout=self.timeout, limits=limits)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def judge_noul(
        self,
        context: str,
        question: str,
        system_instruction: Optional[str] = None,
    ) -> NoulResult:
        client = self._get_client()
        payload = {
            "model": self.model_name,
            "primitive": "noul",
            "context": context,
            "question": question,
            "system_instruction": system_instruction,
        }
        start_time = time.perf_counter()
        try:
            resp = await client.post(self.endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            prob_true = float(data.get("prob_true", 0.5))
            value = bool(data.get("value", prob_true >= 0.5))
            confidence = float(data.get("confidence", max(prob_true, 1.0 - prob_true)))
            latency = (time.perf_counter() - start_time) * 1000.0
            return NoulResult(
                value=value,
                prob_true=prob_true,
                confidence=confidence,
                latency_ms=latency,
                raw=data,
            )
        except Exception as exc:
            latency = (time.perf_counter() - start_time) * 1000.0
            logger.error("Jev vLLM call failed (judge_noul): %s, using safe fallback", exc)
            # Safe fallback: if arbitration fails, default to conservative behavior (require retrieval)
            is_retrieval_question = "needs_external_knowledge" in question.lower()
            return NoulResult(
                value=True if is_retrieval_question else False,
                prob_true=0.9 if is_retrieval_question else 0.1,
                confidence=0.5,
                latency_ms=latency,
                raw={"error": str(exc)},
            )

    async def judge_score(
        self,
        context: str,
        question: str,
        scale_min: float = 1.0,
        scale_max: float = 5.0,
        system_instruction: Optional[str] = None,
    ) -> ScoreResult:
        client = self._get_client()
        payload = {
            "model": self.model_name,
            "primitive": "score",
            "context": context,
            "question": question,
            "parameters": {"scale_min": scale_min, "scale_max": scale_max},
            "system_instruction": system_instruction,
        }
        start_time = time.perf_counter()
        try:
            resp = await client.post(self.endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.perf_counter() - start_time) * 1000.0
            return ScoreResult(
                score=float(data.get("score", scale_min)),
                distribution=data.get("distribution"),
                latency_ms=latency,
                raw=data,
            )
        except Exception as exc:
            latency = (time.perf_counter() - start_time) * 1000.0
            logger.error("Jev vLLM call failed (judge_score): %s, using fallback score", exc)
            # Safe fallback: return middle score
            return ScoreResult(
                score=(scale_min + scale_max) / 2.0,
                latency_ms=latency,
                raw={"error": str(exc)},
            )

    async def judge_choice(
        self,
        context: str,
        question: str,
        options: list[str],
        system_instruction: Optional[str] = None,
    ) -> ChoiceResult:
        client = self._get_client()
        payload = {
            "model": self.model_name,
            "primitive": "choice",
            "context": context,
            "question": question,
            "parameters": {"options": options},
            "system_instruction": system_instruction,
        }
        start_time = time.perf_counter()
        try:
            resp = await client.post(self.endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.perf_counter() - start_time) * 1000.0
            return ChoiceResult(
                selected=data.get("selected", options[0] if options else "unknown"),
                probabilities=data.get("probabilities", {}),
                latency_ms=latency,
                raw=data,
            )
        except Exception as exc:
            latency = (time.perf_counter() - start_time) * 1000.0
            logger.error("Jev vLLM call failed (judge_choice): %s", exc)
            return ChoiceResult(
                selected=options[0] if options else "unknown",
                latency_ms=latency,
                raw={"error": str(exc)},
            )
