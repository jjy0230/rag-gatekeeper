"""Engine factory module."""

from rag_gatekeeper.config import settings
from rag_gatekeeper.engine.interface import DecisionEngine
from rag_gatekeeper.engine.mock_client import MockDecisionEngine
from rag_gatekeeper.engine.jev_client import VllmJevClient


def get_engine() -> DecisionEngine:
    """Return the configured DecisionEngine instance."""
    if settings.USE_MOCK_ENGINE:
        return MockDecisionEngine()
    return VllmJevClient(
        endpoint=settings.JEV_ENDPOINT,
        model_name=settings.JEV_MODEL_NAME,
        timeout_ms=settings.JEV_TIMEOUT_MS,
    )
