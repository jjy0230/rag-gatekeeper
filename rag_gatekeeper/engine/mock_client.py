"""Mock implementation of DecisionEngine for offline development and CI/CD testing."""

import asyncio
import time
from typing import Optional
from rag_gatekeeper.engine.interface import DecisionEngine
from rag_gatekeeper.models import NoulResult, ScoreResult, ChoiceResult


class MockDecisionEngine(DecisionEngine):
    """High-fidelity Mock engine simulating Jev's System One decisions.
    
    Allows developers to verify gatekeeper logic, test edge cases, and run CI
    without setting up local GPU environments or running vllm-jev services.
    """

    def __init__(self, simulated_latency_ms: float = 6.0):
        self.simulated_latency_ms = simulated_latency_ms

    async def _simulate_delay(self):
        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)

    async def judge_noul(
        self,
        context: str,
        question: str,
        system_instruction: Optional[str] = None,
    ) -> NoulResult:
        start_time = time.perf_counter()
        await self._simulate_delay()

        ctx_lower = context.lower()
        q_lower = question.lower()

        # Decision Logic: needs_external_knowledge (Stage 1)
        if "needs_external_knowledge" in q_lower:
            # Common knowledge / greetings / simple generic logic -> False (short-circuit)
            common_patterns = [
                "你好", "hello", "hi", "是谁", "写一首", "1+1", "天气如何",
                "代码怎么写", "快速排序", "定义一个函数", "什么是递归"
            ]
            if any(p in ctx_lower for p in common_patterns) and not any(
                k in ctx_lower for k in ["内部", "公司", "规约", "制度", "文档", "知识库", "业务"]
            ):
                elapsed = (time.perf_counter() - start_time) * 1000.0
                return NoulResult(value=False, prob_true=0.08, confidence=0.92, latency_ms=elapsed)
            
            # Enterprise / internal / specific queries -> True (require retrieval)
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return NoulResult(value=True, prob_true=0.88, confidence=0.88, latency_ms=elapsed)

        # Decision Logic: contains_unsupported_facts (Stage 3)
        if "contains_unsupported_facts" in q_lower:
            # Check for simulated trigger markers in mock mode
            if "__MOCK_HALLUCINATION__" in context or "虚构事实" in context or "伪造凭证" in context:
                elapsed = (time.perf_counter() - start_time) * 1000.0
                return NoulResult(value=True, prob_true=0.85, confidence=0.85, latency_ms=elapsed)
            
            # Default: low hallucination probability (safe)
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return NoulResult(value=False, prob_true=0.05, confidence=0.95, latency_ms=elapsed)

        # Fallback default
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return NoulResult(value=False, prob_true=0.5, confidence=0.5, latency_ms=elapsed)

    async def judge_score(
        self,
        context: str,
        question: str,
        scale_min: float = 1.0,
        scale_max: float = 5.0,
        system_instruction: Optional[str] = None,
    ) -> ScoreResult:
        start_time = time.perf_counter()
        await self._simulate_delay()

        # Score Logic: chunk relevance assessment (Stage 2)
        # Mock calculation: simulate relevance based on token overlap or explicit tags
        ctx_lower = context.lower()
        if "__FORCE_SCORE_" in context:
            # Support deterministic test hooks like __FORCE_SCORE_4.5__
            import re
            match = re.search(r"__FORCE_SCORE_([0-9\.]+)__", context)
            score = float(match.group(1)) if match else 3.5
        elif "无关" in ctx_lower or "广告" in ctx_lower or "免责声明" in ctx_lower:
            score = 1.8
        elif "核心条款" in ctx_lower or "关键信息" in ctx_lower or "标准解答" in ctx_lower:
            score = 4.7
        else:
            # Baseline simulation: default 4.2 for reasonable chunks
            score = 4.2

        score = max(scale_min, min(scale_max, score))
        elapsed = (time.perf_counter() - start_time) * 1000.0

        return ScoreResult(
            score=score,
            distribution={1: 0.05, 2: 0.1, 3: 0.2, 4: 0.35, 5: 0.3},
            latency_ms=elapsed,
        )

    async def judge_choice(
        self,
        context: str,
        question: str,
        options: list[str],
        system_instruction: Optional[str] = None,
    ) -> ChoiceResult:
        start_time = time.perf_counter()
        await self._simulate_delay()

        ctx_lower = context.lower()
        selected = options[0] if options else "unknown"
        probs = {opt: 0.1 for opt in options}

        # Simulated semantic choice for domain routing
        if any(k in ctx_lower for k in ["报销", "发票", "住宿", "财报", "补贴", "交通"]) and "FINANCE_EXPENSE" in options:
            selected = "FINANCE_EXPENSE"
            probs["FINANCE_EXPENSE"] = 0.88
        elif any(k in ctx_lower for k in ["年假", "离职", "入职", "福利", "社保", "考勤"]) and "HR_POLICY" in options:
            selected = "HR_POLICY"
            probs["HR_POLICY"] = 0.90
        elif any(k in ctx_lower for k in ["vpn", "代码", "服务器", "系统", "工单"]) and "IT_DEVOPS" in options:
            selected = "IT_DEVOPS"
            probs["IT_DEVOPS"] = 0.85
        elif any(k in ctx_lower for k in ["合同", "保密", "协议", "法务", "nda"]) and "LEGAL_CONTRACT" in options:
            selected = "LEGAL_CONTRACT"
            probs["LEGAL_CONTRACT"] = 0.87
        elif options:
            probs[options[0]] = 0.70

        elapsed = (time.perf_counter() - start_time) * 1000.0
        return ChoiceResult(
            selected=selected,
            probabilities=probs,
            latency_ms=elapsed,
        )
