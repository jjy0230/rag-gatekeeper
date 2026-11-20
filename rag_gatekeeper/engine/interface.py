"""Abstract interface for Jev / System One decision engines."""

from abc import ABC, abstractmethod
from typing import Optional
from rag_gatekeeper.models import NoulResult, ScoreResult, ChoiceResult


class DecisionEngine(ABC):
    """Abstract interface defining the contract for Jev decision primitives.
    
    Any underlying backend (mode-io/vllm-jev HTTP endpoint, local in-process vLLM,
    or Mock engine for unit testing) must implement this interface.
    """

    @abstractmethod
    async def judge_noul(
        self,
        context: str,
        question: str,
        system_instruction: Optional[str] = None,
    ) -> NoulResult:
        """Evaluate a calibrated Boolean (Yes/No) decision with probability.
        
        Args:
            context: The text state / context (e.g. user query or [answer + chunks]).
            question: The typed boolean question (e.g. 'needs_external_knowledge').
            system_instruction: Optional prefix instruction (benefits from KV prefix caching).
            
        Returns:
            NoulResult containing boolean value, confidence, and latency.
        """
        pass

    @abstractmethod
    async def judge_score(
        self,
        context: str,
        question: str,
        scale_min: float = 1.0,
        scale_max: float = 5.0,
        system_instruction: Optional[str] = None,
    ) -> ScoreResult:
        """Evaluate a rating score within a specified range.
        
        Args:
            context: Text input (e.g. [Query: ... \n Chunk: ...]).
            question: The rating dimension (e.g. 'relevance_to_query').
            scale_min: Lower bound (default: 1.0).
            scale_max: Upper bound (default: 5.0).
            system_instruction: Optional prefix instruction.
            
        Returns:
            ScoreResult containing calculated score and probability distribution.
        """
        pass

    @abstractmethod
    async def judge_choice(
        self,
        context: str,
        question: str,
        options: list[str],
        system_instruction: Optional[str] = None,
    ) -> ChoiceResult:
        """Select among categorical options.
        
        Args:
            context: Text context.
            question: The categorization question.
            options: Set of discrete candidate options.
            system_instruction: Optional prefix instruction.
            
        Returns:
            ChoiceResult containing chosen option and categorical probabilities.
        """
        pass
