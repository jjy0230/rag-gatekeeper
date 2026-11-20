"""Comprehensive test suite for RAG-Gatekeeper."""

import pytest
import httpx
from rag_gatekeeper.config import settings
from rag_gatekeeper.matcher import AhoCorasickMatcher
from rag_gatekeeper.engine.mock_client import MockDecisionEngine
from rag_gatekeeper.stages.stage1_arbitration import RetrievalArbitrator
from rag_gatekeeper.stages.stage2_rerank import ChunkReranker
from rag_gatekeeper.stages.stage3_guardrail import HallucinationGuardrail
from rag_gatekeeper.pipeline import RAGGatekeeperPipeline
from rag_gatekeeper.server import app
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


def test_aho_corasick_matcher():
    """Verify Aho-Corasick automaton matching accuracy and performance."""
    keywords = ["内部规范", "SLA", "差旅报销", "Q3财报"]
    matcher = AhoCorasickMatcher(keywords)

    assert matcher.has_any("请问关于内部规范的修改通知") is True
    assert matcher.has_any("SLA要求多少？") is True
    assert matcher.has_any("完全不相干的公开问题") is False
    assert set(matcher.search_all("请参考SLA和内部规范")) == {"SLA", "内部规范"}


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


@pytest.mark.asyncio
async def test_api_server_proxy_and_metrics():
    """Verify FastAPI server proxy (non-stream & SSE stream) and metrics endpoints."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Healthz
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

        # 2. Non-stream chat completion
        body_non_stream = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "你好，写一个hello world"}],
            "stream": False,
        }
        res_ns = await client.post("/v1/chat/completions", json=body_non_stream)
        assert res_ns.status_code == 200
        assert "X-Gatekeeper-Short-Circuited" in res_ns.headers
        data = res_ns.json()
        assert "choices" in data
        assert "gatekeeper" in data

        # 3. Stream chat completion
        body_stream = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "公司差旅报销政策"}],
            "stream": True,
        }
        res_stream = await client.post("/v1/chat/completions", json=body_stream)
        assert res_stream.status_code == 200
        assert "text/event-stream" in res_stream.headers["content-type"]
        text_stream = res_stream.text
        assert "data: " in text_stream
        assert "[DONE]" in text_stream
        assert "gatekeeper_audit" in text_stream

        # 4. Metrics endpoint
        res_metrics = await client.get("/metrics")
        assert res_metrics.status_code == 200
        assert "rag_gk_requests_total" in res_metrics.text

        # 5. Dynamic config reload
        reload_payload = {
            "FORCE_RETRIEVAL_KEYWORDS": ["特急工单", "CEO审批"],
            "RERANK_SCORE_THRESHOLD": 3.8,
        }
        res_reload = await client.post("/v1/gatekeeper/config/reload", json=reload_payload)
        assert res_reload.status_code == 200
        assert res_reload.json()["status"] == "reloaded"
        assert settings.RERANK_SCORE_THRESHOLD == 3.8
