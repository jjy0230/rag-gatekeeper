"""Adaptive Tiered Confidence Radar & Disclaimer Manager.

Eliminates binary all-or-nothing blocking by classifying answers into 3 confidence tiers:
  Tier 1 (>= 0.85): Authoritative Verified (green light)
  Tier 2 (0.50 ~ 0.85): Partially Grounded (yellow light, injects targeted operational disclaimer)
  Tier 3 (< 0.50): Hallucinated Block (red light, blocks and degrades)
"""

from typing import Literal
from pydantic import BaseModel, Field


class ConfidenceTierResult(BaseModel):
    tier: Literal["AUTHORITATIVE_VERIFIED", "PARTIALLY_GROUNDED", "HALLUCINATED_BLOCKED"]
    confidence_score: float
    hallucination_prob: float
    final_output: str
    action_taken: str
    disclaimer_added: bool


class ConfidenceRadar:
    """Evaluates factuality confidence and applies tiered adaptive disclaimers."""

    DEFAULT_DISCLAIMER = (
        "\n\n> ⚠️ **【事实边界指引】** 核心政策已依据检索文档核验一致；"
        "回答中关于流程细节的衍生说明未见明确书面条款，请以对应主管部门实际审批答复为准。"
    )

    def __init__(
        self,
        high_threshold: float = 0.85,
        low_threshold: float = 0.50,
        fallback_template: str = "【安全拦截】回答缺乏可靠依据，为保障信息准确已阻断。",
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.fallback_template = fallback_template

    def evaluate_tier(
        self,
        answer: str,
        hallucination_prob: float,
    ) -> ConfidenceTierResult:
        """Categorize answer into 3-tier radar and append adaptive disclaimer if needed."""
        # Grounding confidence is inverted hallucination probability
        confidence = 1.0 - hallucination_prob

        # Tier 3: High hallucination -> Block
        if hallucination_prob >= (1.0 - self.low_threshold) or confidence < self.low_threshold:
            return ConfidenceTierResult(
                tier="HALLUCINATED_BLOCKED",
                confidence_score=round(confidence, 3),
                hallucination_prob=round(hallucination_prob, 3),
                final_output=self.fallback_template,
                action_taken="blocked_and_degraded",
                disclaimer_added=False,
            )

        # Tier 2: Partially grounded -> Pass with disclaimer
        if confidence < self.high_threshold:
            annotated = answer.strip() + self.DEFAULT_DISCLAIMER
            return ConfidenceTierResult(
                tier="PARTIALLY_GROUNDED",
                confidence_score=round(confidence, 3),
                hallucination_prob=round(hallucination_prob, 3),
                final_output=annotated,
                action_taken="annotated_with_disclaimer",
                disclaimer_added=True,
            )

        # Tier 1: Authoritative Verified -> Pure pass
        return ConfidenceTierResult(
            tier="AUTHORITATIVE_VERIFIED",
            confidence_score=round(confidence, 3),
            hallucination_prob=round(hallucination_prob, 3),
            final_output=answer,
            action_taken="passed_authoritative",
            disclaimer_added=False,
        )


# Global singleton radar
confidence_radar = ConfidenceRadar()
