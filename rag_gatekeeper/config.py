"""Configuration settings for RAG-Gatekeeper gateway."""

from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatekeeperConfig(BaseSettings):
    """Runtime configuration for RAG-Gatekeeper."""

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    DEBUG: bool = False

    # Decision Engine backend settings
    # Default is Mock mode so users can verify the logic instantly without local GPU deployment.
    USE_MOCK_ENGINE: bool = True
    JEV_ENDPOINT: str = "http://localhost:8000/v1/systemone"
    JEV_MODEL_NAME: str = "mode-io/vllm-jev"
    JEV_TIMEOUT_MS: int = 150
    JEV_MAX_RETRIES: int = 2

    # Stage 1: Retrieval Arbitration Gate
    ENABLE_STAGE1_ARBITRATION: bool = True
    ARBITRATION_CONFIDENCE_THRESHOLD: float = 0.5
    # Bypass keywords or patterns that always force retrieval (e.g. enterprise domain terms)
    FORCE_RETRIEVAL_KEYWORDS: list[str] = ["政策", "内部规范", "财报", "工单", "审批", "SLA", "Q1", "Q2", "Q3", "Q4"]

    # Stage 2: Lightweight Reranker & Noise Filtering
    ENABLE_STAGE2_RERANK: bool = True
    RERANK_SCORE_THRESHOLD: float = 4.0  # Cutoff score [1.0 - 5.0]
    RERANK_MAX_CHUNK_CHARS: int = 300   # Pre-truncation to prevent long prefill dilution
    RERANK_CONCURRENCY: int = 16        # Max parallel chunk scoring requests
    # Anti-Recall-Cliff Safeguard: if all chunks < threshold, fallback to top N
    RERANK_FALLBACK_TOP_K: int = 2

    # Stage 3: Hallucination & Factuality Gatekeeper
    ENABLE_STAGE3_GUARDRAIL: bool = True
    HALLUCINATION_PROB_THRESHOLD: float = 0.3  # Probability > 0.3 triggers block
    GUARDRAIL_MODE: Literal["BLOCK_AND_DEGRADE", "FLAG_ONLY"] = "BLOCK_AND_DEGRADE"
    FALLBACK_TEMPLATE: str = (
        "【安全拦截】经事实一致性校验，大模型生成的回答缺乏检索文档切片的直接支撑（幻觉置信度超标）。"
        "为保障信息准确性，已阻断输出。请尝试补充上下文或调整提问范围。"
    )

    # Upstream LLM backend (for transparent OpenAI proxying)
    UPSTREAM_LLM_URL: Optional[str] = None
    UPSTREAM_LLM_API_KEY: str = "EMPTY"
    UPSTREAM_TIMEOUT_SEC: float = 60.0

    model_config = SettingsConfigDict(
        env_prefix="RAG_GK_",
        env_file=".env",
        extra="ignore",
    )


# Global settings singleton
settings = GatekeeperConfig()


def reload_settings(updates: dict) -> GatekeeperConfig:
    """Dynamically update settings at runtime."""
    for key, val in updates.items():
        if hasattr(settings, key):
            setattr(settings, key, val)
    return settings
