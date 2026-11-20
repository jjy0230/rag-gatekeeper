"""Speculative Sentence-Level Guardrail for streaming Early-Stopping."""

import re
import time
import logging
from typing import Optional, AsyncGenerator
from rag_gatekeeper.engine.interface import DecisionEngine
from rag_gatekeeper.models import ChunkItem, GuardrailRequest

logger = logging.getLogger(__name__)

# Sentence boundary delimiter (Chinese & English punctuation + newlines)
SENTENCE_SPLIT_REGEX = re.compile(r"([。！？\n!?]+)")


class SpeculativeSentenceGuardrail:
    """Evaluates factual grounding sentence-by-sentence during LLM token streaming.
    
    If any single sentence exhibits high hallucination probability (> threshold),
    triggers early stopping, cuts downstream SSE streaming, and yields an override frame.
    """

    def __init__(self, engine: DecisionEngine, early_kill_threshold: float = 0.75):
        self.engine = engine
        self.early_kill_threshold = early_kill_threshold

    def split_sentences(self, buffer: str) -> tuple[list[str], str]:
        """Split buffered stream into completed sentences and leftover incomplete fragment."""
        parts = SENTENCE_SPLIT_REGEX.split(buffer)
        sentences = []
        # Reconstruct sentence with its delimiter
        for i in range(0, len(parts) - 1, 2):
            sent = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
            if sent.strip():
                sentences.append(sent)
        # Last element is the incomplete fragment
        leftover = parts[-1] if len(parts) % 2 == 1 else ""
        return sentences, leftover

    async def verify_sentence(
        self,
        query: str,
        sentence: str,
        evidence_chunks: list[str],
    ) -> tuple[bool, float]:
        """Verify grounding of a single sentence.
        
        Returns:
            (is_hallucinated, hallucination_prob)
        """
        if not evidence_chunks or not sentence.strip():
            return False, 0.0

        formatted_evidence = "\n".join([f"[{i+1}]: {c}" for i, c in enumerate(evidence_chunks)])
        context = (
            f"Evidence Context:\n{formatted_evidence}\n\n"
            f"Query: {query}\n"
            f"Generated Sentence to Validate: {sentence.strip()}"
        )
        system_instruction = (
            "Verify whether this generated sentence contains factual claims unsupported by the evidence. "
            "Return True if unsupported facts or contradictions exist, False if grounded."
        )

        noul_res = await self.engine.judge_noul(
            context=context,
            question="contains_unsupported_facts",
            system_instruction=system_instruction,
        )

        prob = noul_res.prob_true
        is_hallucinated = prob >= self.early_kill_threshold
        return is_hallucinated, prob
