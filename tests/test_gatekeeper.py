"""Comprehensive test suite for RAG-Gatekeeper."""

import pytest
import httpx
from rag_gatekeeper.config import settings
from rag_gatekeeper.matcher import AhoCorasickMatcher
from rag_gatekeeper.engine.mock_client import MockDecisionEngine
from rag_gatekeeper.stages.stage1_arbitration import RetrievalArbitrator
from rag_gatekeeper.stages.stage2_rerank import ChunkReranker
from rag_gatekeeper.stages.stage3_guardrail import HallucinationGuardrail
from rag_gatekeeper.speculative_guardrail import SpeculativeSentenceGuardrail
from rag_gatekeeper.vector_store import InMemoryHybridStore
from rag_gatekeeper.audit_sink import audit_sink
from rag_gatekeeper.semantic_cache import semantic_cache
from rag_gatekeeper.citation_injector import citation_injector
from rag_gatekeeper.context_compressor import context_compressor
from rag_gatekeeper.query_rewriter import query_rewriter
from rag_gatekeeper.domain_router import DomainRouter, KnowledgeDomain
from rag_gatekeeper.confidence_radar import confidence_radar
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
    """Verify end-to-end pipeline flow across all stages with citations."""
    pipeline = RAGGatekeeperPipeline(mock_engine)

    # 1. Pipeline Short-circuit test
    short_res = await pipeline.execute(PipelineRequest(query="1+1在二进制里怎么写？"))
    assert short_res.short_circuited is True
    assert short_res.rerank is None

    # 2. Pipeline Retrieval + Rerank + Guardrail + Footnotes Citation test
    chunks = [
        ChunkItem(id="c1", content="核心条款：年假未休可折算补贴。", metadata={"title": "年假规程"}),
        ChunkItem(id="c2", content="无关广告信息。"),
    ]
    pipe_res = await pipeline.execute(
        PipelineRequest(
            query="内部政策：年假没休完怎么办？",
            candidate_chunks=chunks,
            generated_answer="未休完的年假可在年底按规定折算补贴。",
        )
    )
    assert pipe_res.short_circuited is False
    assert pipe_res.rerank.kept_count == 1
    assert pipe_res.guardrail.blocked is False
    assert "[1]" in pipe_res.final_output
    assert "参考来源：" in pipe_res.final_output
    assert pipe_res.confidence_tier == "AUTHORITATIVE_VERIFIED"


@pytest.mark.asyncio
async def test_domain_router_choice(mock_engine):
    """Verify Jev Choice multi-domain routing for HR and Finance."""
    router = DomainRouter(mock_engine)
    res_fin = await router.route_query("差旅发票住宿每天报销上限")
    assert res_fin.target_domain in ["FINANCE_EXPENSE", "HR_POLICY", "IT_DEVOPS", "LEGAL_CONTRACT"]
    assert res_fin.confidence > 0.5


def test_confidence_radar():
    """Verify 3-tier confidence radar (Authoritative, Partial with disclaimer, Blocked)."""
    # Tier 1: Authoritative (prob=0.05 -> conf=0.95)
    t1 = confidence_radar.evaluate_tier("员工守则严禁携带宠物。", hallucination_prob=0.05)
    assert t1.tier == "AUTHORITATIVE_VERIFIED"
    assert t1.disclaimer_added is False

    # Tier 2: Partially grounded (prob=0.30 -> conf=0.70)
    t2 = confidence_radar.evaluate_tier("差旅每日补贴100元，建议找财务审批。", hallucination_prob=0.30)
    assert t2.tier == "PARTIALLY_GROUNDED"
    assert t2.disclaimer_added is True
    assert "事实边界指引" in t2.final_output

    # Tier 3: Blocked (prob=0.85 -> conf=0.15)
    t3 = confidence_radar.evaluate_tier("虚构捏造事实。", hallucination_prob=0.85)
    assert t3.tier == "HALLUCINATED_BLOCKED"
    assert "安全拦截" in t3.final_output


def test_partitioned_vector_search():
    """Verify vector search partition key filtering."""
    store = InMemoryHybridStore()
    docs = [
        ChunkItem(id="hr_1", content="HR年假规定", metadata={"partition_key": "hr"}),
        ChunkItem(id="fin_1", content="财务报销规定", metadata={"partition_key": "finance"}),
    ]
    store.upsert_chunks(docs)

    res_hr = store.search("规定", top_k=5, partition_key="hr")
    assert len(res_hr) == 1
    assert res_hr[0].id == "hr_1"


@pytest.mark.asyncio
async def test_speculative_sentence_guardrail(mock_engine):
    """Verify speculative sentence-level guardrail detects and flags early hallucinations."""
    guard = SpeculativeSentenceGuardrail(mock_engine, early_kill_threshold=0.75)
    
    # 1. Test sentence splitting
    text_stream = "这是第一句话。这是第二句话！还有未完成的一半"
    sentences, leftover = guard.split_sentences(text_stream)
    assert len(sentences) == 2
    assert sentences[0] == "这是第一句话。"
    assert sentences[1] == "这是第二句话！"
    assert leftover == "还有未完成的一半"

    # 2. Test sentence hallucination check
    is_hallu, prob = await guard.verify_sentence(
        query="公司福利",
        sentence="我们每个月发一万块宠物补助。__MOCK_HALLUCINATION__",
        evidence_chunks=["公司只提供午餐补贴。"],
    )
    assert is_hallu is True
    assert prob >= 0.75


def test_in_memory_hybrid_vector_store():
    """Verify native in-memory hybrid store (BM25 + Dense RRF)."""
    store = InMemoryHybridStore()
    docs = [
        ChunkItem(id="d1", content="员工差旅补贴：每日交通补助 100 元，住宿凭票实报实销。"),
        ChunkItem(id="d2", content="员工休假制度：年假每年五天，未休完可折算。"),
        ChunkItem(id="d3", content="公司前台电话与访客登记规范。"),
    ]
    store.upsert_chunks(docs)

    results = store.search("差旅交通补贴多少钱？", top_k=2)
    assert len(results) == 2
    assert results[0].id == "d1"


def test_semantic_cache_l1_and_l2():
    """Verify semantic cache Exact L1 and Cosine Semantic L2 hits."""
    semantic_cache.clear()
    semantic_cache.set("年假未休完怎么处理？", "年假未休完可在年底折算补贴发放。")

    # 1. Exact match (L1)
    res_l1 = semantic_cache.get("年假未休完怎么处理？")
    assert res_l1 is not None
    assert "折算补贴" in res_l1["response"]

    # 2. Semantic match (L2 equivalent wording)
    res_l2 = semantic_cache.get("如果年假没休完怎么处理呢")
    assert res_l2 is not None
    assert "折算补贴" in res_l2["response"]

    # 3. Completely irrelevant -> Miss
    res_miss = semantic_cache.get("明天天气如何")
    assert res_miss is None


def test_citation_footnotes_injector():
    """Verify citation marker insertion and footnote generation."""
    chunks = [
        ChunkItem(id="c1", content="差旅每天享有 100 元交通补贴。", metadata={"title": "差旅报销办法"}),
        ChunkItem(id="c2", content="年假每年固定五天，未休完可转下一年。", metadata={"title": "员工休假规范"}),
    ]
    answer = "员工出差每天可报销 100 元交通费。此外年假共有五天。"
    res = citation_injector.inject_citations(answer, chunks)
    assert res.citations_count >= 1
    assert "[1]" in res.annotated_answer
    assert "**参考来源：**" in res.annotated_answer
    assert "差旅报销办法" in res.annotated_answer


def test_context_compressor():
    """Verify extractive sentence compressor extracts salient sentences and drops fluff."""
    chunk = ChunkItem(
        id="c1",
        content="公司简介：本公司成立于2010年，秉承诚信原则。差旅补贴标准为每日100元交通补助。欢迎广大客户监督合作。",
    )
    compressed = context_compressor.compress_chunk("差旅每天补贴多少", chunk)
    assert "差旅补贴标准为每日100元交通补助" in compressed.content
    assert compressed.metadata["compressed"] is True
    assert compressed.metadata["compressed_len"] < compressed.metadata["original_len"]


def test_query_rewriter():
    """Verify conversational pronoun de-referencing."""
    # With pronoun
    rewritten, was_rw = query_rewriter.rewrite_query("它的上限是多少？", last_turn_query="差旅交通补贴规定")
    assert was_rw is True
    assert "差旅交通补贴规定" in rewritten

    # Without pronoun (standalone)
    standalone, was_rw2 = query_rewriter.rewrite_query("什么是快速排序？", last_turn_query="差旅交通补贴规定")
    assert was_rw2 is False
    assert standalone == "什么是快速排序？"


@pytest.mark.asyncio
async def test_audit_sink_and_export():
    """Verify audit event logging and Jev dataset export."""
    await audit_sink.log_event("STAGE1_ARBITRATION", {
        "query": "你好",
        "needs_external_knowledge": False,
        "confidence": 0.95,
    })
    count = audit_sink.export_jev_finetuning_dataset()
    assert count >= 0


@pytest.mark.asyncio
async def test_api_server_proxy_and_metrics():
    """Verify FastAPI server proxy (non-stream & SSE stream) and metrics endpoints."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Healthz
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

        # 2. Domains list
        res_doms = await client.get("/v1/gatekeeper/domains")
        assert res_doms.status_code == 200
        assert len(res_doms.json()) >= 4

        # 3. Non-stream chat completion
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

        # 4. Stream chat completion
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

        # 5. Metrics endpoint
        res_metrics = await client.get("/metrics")
        assert res_metrics.status_code == 200
        assert "rag_gk_requests_total" in res_metrics.text

        # 6. Dynamic config reload
        reload_payload = {
            "FORCE_RETRIEVAL_KEYWORDS": ["特急工单", "CEO审批"],
            "RERANK_SCORE_THRESHOLD": 3.8,
        }
        res_reload = await client.post("/v1/gatekeeper/config/reload", json=reload_payload)
        assert res_reload.status_code == 200
        assert res_reload.json()["status"] == "reloaded"
        assert settings.RERANK_SCORE_THRESHOLD == 3.8

        # 7. Dashboard HTML endpoint
        res_dash = await client.get("/dashboard")
        assert res_dash.status_code == 200
        assert "RAG-Gatekeeper" in res_dash.text

        # 8. Interactive Demo HTML endpoint
        res_demo = await client.get("/demo")
        assert res_demo.status_code == 200
        assert "全景决策网关交互演示" in res_demo.text
