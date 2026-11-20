"""Comprehensive test suite for RAG-Gatekeeper."""

import pytest
from rag_gatekeeper.config import settings
from rag_gatekeeper.engine.mock_client import MockDecisionEngine
from rag_gatekeeper.stages.stage1_arbitration import RetrievalArbitrator
from rag_gatekeeper.stages.stage2_rerank import ChunkReranker
from rag_gatekeeper.stages.stage3_guardrail import HallucinationGuardrail
from rag_gatekeeper.pipeline import RAGGatekeeperPipeline
from rag_gatekeeper.models import (
    ArbitrationRequest,
    ChunkItem,
    RerankRequest,
    GuardrailRequest,
    PipelineRequest,
)


@pytest.fixture
def mock_engine():
    # Zero simulated delay for instant tests
    return MockDecisionEngine(simulated_latency_ms=0.0)


@pytest.mark.asyncio
async def test_stage1_short_circuit(mock_engine):
    """Verify trivial greeting or common sense query is short-circuited (needs_retrieval=False)."""
    stage1 = RetrievalArbitrator(mock_engine)
    res = await stage1.arbitrate(ArbitrationRequest(query="你好，请问快速排序的时间复杂度是多少？"))
    assert res.needs_external_knowledge is False
    assert res.confidence > 0.8
    assert res.reasoning_mode == "jev_noul"


@pytest.mark.asyncio
async def test_stage1_require_retrieval(mock_engine):
    """Verify domain enterprise query triggers retrieval (needs_retrieval=True)."""
    stage1 = RetrievalArbitrator(mock_engine)
    res = await stage1.arbitrate(ArbitrationRequest(query="请帮我查询我们部门最新的差旅报销审批标准"))
    assert res.needs_external_knowledge is True
    assert res.confidence >= 0.8


@pytest.mark.asyncio
async def test_stage1_keyword_bypass(mock_engine):
    """Verify fast rule bypass on explicit enterprise SLA keywords."""
    stage1 = RetrievalArbitrator(mock_engine)
    res = await stage1.arbitrate(ArbitrationRequest(query="Q3内部规范修改内容是什么"))
    assert res.needs_external_knowledge is True
    assert res.reasoning_mode == "keyword_bypass"
    assert res.confidence == 1.0


@pytest.mark.asyncio
async def test_stage2_rerank_and_filter(mock_engine):
    """Verify chunks with low relevance (<4.0) are filtered out, high relevance kept."""
    stage2 = ChunkReranker(mock_engine)
    chunks = [
        ChunkItem(id="c1", content="核心条款：差旅每天交通补贴为 100 元"),  # Mock returns 4.7
        ChunkItem(id="c2", content="广告与无关免责声明"),  # Mock returns 1.8
        ChunkItem(id="c3", content="标准解答：住宿发票必须在 7 个工作日内提交"),  # Mock returns 4.7
    ]
    res = await stage2.rerank(RerankRequest(query="差旅补贴怎么报销", chunks=chunks))
    assert res.total_evaluated == 3
    assert res.kept_count == 2
    assert [sc.chunk.id for sc in res.filtered_chunks] == ["c1", "c3"]
    assert res.fallback_applied is False


@pytest.mark.asyncio
async def test_stage2_anti_recall_cliff_fallback(mock_engine):
    """Verify anti-recall-cliff safeguard retains top K chunks when all score below threshold."""
    stage2 = ChunkReranker(mock_engine)
    # Both chunks are low scoring (< 4.0)
    chunks = [
        ChunkItem(id="c1", content="无关免责声明一"),
        ChunkItem(id="c2", content="无关免责声明二"),
        ChunkItem(id="c3", content="广告推广信息三"),
    ]
    res = await stage2.rerank(RerankRequest(query="某冷门提问", chunks=chunks))
    # Safeguard should activate and retain top 2
    assert res.fallback_applied is True
    assert res.kept_count == 2


@pytest.mark.asyncio
async def test_stage3_hallucination_block(mock_engine):
    """Verify ungrounded/hallucinated answer is blocked and replaced by fallback template."""
    stage3 = HallucinationGuardrail(mock_engine)
    req = GuardrailRequest(
        query="公司允许带宠物上班吗？",
        answer="允许，并且提供猫粮补助。__MOCK_HALLUCINATION__",
        evidence_chunks=["员工守则第三条：严禁私自携带宠物进入办公区域。"],
    )
    res = await stage3.verify(req)
    assert res.blocked is True
    assert res.degraded is True
    assert res.final_answer == settings.FALLBACK_TEMPLATE


@pytest.mark.asyncio
async def test_stage3_grounded_pass(mock_engine):
    """Verify fully grounded factual answer is passed through cleanly."""
    stage3 = HallucinationGuardrail(mock_engine)
    safe_answer = "根据员工守则，公司严禁携带宠物进入办公区域。"
    req = GuardrailRequest(
        query="公司允许带宠物上班吗？",
        answer=safe_answer,
        evidence_chunks=["员工守则第三条：严禁私自携带宠物进入办公区域。"],
    )
    res = await stage3.verify(req)
    assert res.blocked is False
    assert res.degraded is False
    assert res.final_answer == safe_answer


@pytest.mark.asyncio
async def test_pipeline_orchestration(mock_engine):
    """Verify end-to-end pipeline flow across all stages."""
    pipeline = RAGGatekeeperPipeline(mock_engine)

    # 1. Pipeline Short-circuit test
    short_res = await pipeline.execute(PipelineRequest(query="1+1在二进制里怎么写？"))
    assert short_res.short_circuited is True
    assert short_res.rerank is None

    # 2. Pipeline Retrieval + Rerank + Guardrail test
    chunks = [
        ChunkItem(id="c1", content="核心条款：年假未休可折算补贴"),
        ChunkItem(id="c2", content="无关广告信息"),
    ]
    pipe_res = await pipeline.execute(
        PipelineRequest(
            query="内部政策：年假没休完怎么办？",
            candidate_chunks=chunks,
            generated_answer="未休完的年假可在年底按规定折算补贴。",
        )
    )
    assert pipe_res.short_circuited is False
    assert pipe_res.rerank is not None
    assert pipe_res.rerank.kept_count == 1
    assert pipe_res.guardrail is not None
    assert pipe_res.guardrail.blocked is False
    assert pipe_res.total_overhead_latency_ms >= 0.0
