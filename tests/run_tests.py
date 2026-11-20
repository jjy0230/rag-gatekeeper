"""Standard library runner for RAG-Gatekeeper tests (no external test runner required)."""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

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
from rag_gatekeeper.domain_router import DomainRouter
from rag_gatekeeper.confidence_radar import confidence_radar
from rag_gatekeeper.pipeline import RAGGatekeeperPipeline
from rag_gatekeeper.models import (
    ArbitrationRequest,
    ChunkItem,
    RerankRequest,
    GuardrailRequest,
    PipelineRequest,
)


async def main():
    print("========================================")
    print("Running RAG-Gatekeeper Full Verification (Final Suite)")
    print("========================================")
    engine = MockDecisionEngine(simulated_latency_ms=0.0)

    # Test 0: Aho-Corasick Matcher
    print("[1/16] Testing Aho-Corasick Multi-Pattern Matcher...")
    matcher = AhoCorasickMatcher(["内部规范", "SLA", "差旅报销"])
    assert matcher.has_any("内部规范变更通知") is True
    assert matcher.has_any("公开常识问答") is False
    print("       ✓ Aho-Corasick Automaton passed.")

    # Test 1: Stage 1 Short Circuit
    print("[2/16] Testing Stage 1 Short-circuiting...")
    stage1 = RetrievalArbitrator(engine)
    res1 = await stage1.arbitrate(ArbitrationRequest(query="你好，请问快速排序的时间复杂度是多少？"))
    assert res1.needs_external_knowledge is False, f"Expected False, got {res1}"
    print("       ✓ Stage 1 Short-circuit passed.")

    # Test 2: Stage 1 Retrieval Trigger
    print("[3/16] Testing Stage 1 Knowledge Retrieval Trigger...")
    res2 = await stage1.arbitrate(ArbitrationRequest(query="请帮我查询我们部门最新的差旅报销审批标准"))
    assert res2.needs_external_knowledge is True, f"Expected True, got {res2}"
    print("       ✓ Stage 1 Trigger passed.")

    # Test 3: Stage 2 Rerank & Noise Filtering
    print("[4/16] Testing Stage 2 Reranking & Filtering...")
    stage2 = ChunkReranker(engine)
    chunks = [
        ChunkItem(id="c1", content="核心条款：差旅每天交通补贴为 100 元"),
        ChunkItem(id="c2", content="广告与无关免责声明"),
        ChunkItem(id="c3", content="标准解答：住宿发票必须在 7 个工作日内提交"),
    ]
    res3 = await stage2.rerank(RerankRequest(query="差旅补贴怎么报销", chunks=chunks))
    assert res3.kept_count == 2, f"Expected 2 kept chunks, got {res3.kept_count}"
    assert [sc.chunk.id for sc in res3.filtered_chunks] == ["c1", "c3"]
    print("       ✓ Stage 2 Noise Filtering passed.")

    # Test 4: Stage 2 Anti-Recall-Cliff Fallback
    print("[5/16] Testing Stage 2 Anti-Recall-Cliff Fallback...")
    low_chunks = [
        ChunkItem(id="c1", content="无关免责声明一"),
        ChunkItem(id="c2", content="无关免责声明二"),
    ]
    res4 = await stage2.rerank(RerankRequest(query="冷门问答", chunks=low_chunks))
    assert res4.fallback_applied is True, "Expected fallback_applied=True"
    assert res4.kept_count == 2, f"Expected 2 chunks retained in fallback, got {res4.kept_count}"
    print("       ✓ Stage 2 Fallback Safeguard passed.")

    # Test 5: Stage 3 Hallucination Block
    print("[6/16] Testing Stage 3 Hallucination Blocking...")
    stage3 = HallucinationGuardrail(engine)
    req5 = GuardrailRequest(
        query="公司允许带宠物上班吗？",
        answer="允许，并且提供猫粮补助。__MOCK_HALLUCINATION__",
        evidence_chunks=["员工守则第三条：严禁私自携带宠物进入办公区域。"],
    )
    res5 = await stage3.verify(req5)
    assert res5.blocked is True, "Expected hallucination to be blocked"
    print("       ✓ Stage 3 Hallucination Block passed.")

    # Test 6: Domain Router Choice
    print("[7/16] Testing Domain Router Choice Primitive...")
    router = DomainRouter(engine)
    res_dom = await router.route_query("差旅发票住宿报销上限")
    assert res_dom.target_domain == "FINANCE_EXPENSE"
    assert res_dom.confidence >= 0.8
    print("       ✓ Domain Router Choice passed.")

    # Test 7: 3-Tier Confidence Radar
    print("[8/16] Testing 3-Tier Confidence Radar...")
    t_partial = confidence_radar.evaluate_tier("差旅补贴100元，建议审批确认。", hallucination_prob=0.30)
    assert t_partial.tier == "PARTIALLY_GROUNDED"
    assert t_partial.disclaimer_added is True
    print("       ✓ Confidence Radar passed.")

    # Test 8: Partitioned Vector Store
    print("[9/16] Testing Partitioned Vector Store Isolation...")
    store = InMemoryHybridStore()
    store.upsert_chunks([
        ChunkItem(id="p1", content="财务发票规定", metadata={"partition_key": "finance"}),
        ChunkItem(id="p2", content="HR年假规定", metadata={"partition_key": "hr"}),
    ])
    matched = store.search("规定", top_k=5, partition_key="finance")
    assert len(matched) == 1 and matched[0].id == "p1"
    print("       ✓ Partitioned Store passed.")

    # Test 9: Speculative Sentence Guardrail
    print("[10/16] Testing Speculative Sentence Guardrail...")
    s_guard = SpeculativeSentenceGuardrail(engine, early_kill_threshold=0.75)
    sentences, leftover = s_guard.split_sentences("第一句。第二句！未完")
    assert len(sentences) == 2
    is_h, p = await s_guard.verify_sentence("福利", "发一万。__MOCK_HALLUCINATION__", ["无此福利"])
    assert is_h is True
    print("        ✓ Speculative Sentence Guardrail passed.")

    # Test 10: Semantic Cache L1 & L2
    print("[11/16] Testing Semantic Cache...")
    semantic_cache.clear()
    semantic_cache.set("年假未休完怎么处理？", "年假未休完可在年底折算补贴发放。")
    assert semantic_cache.get("如果年假没休完怎么处理呢") is not None
    print("        ✓ Semantic Cache passed.")

    # Test 11: Citation Footnotes Injector
    print("[12/16] Testing Citation & Footnote Injector...")
    cite_c = [ChunkItem(id="c1", content="差旅每天享有 100 元交通补贴。", metadata={"title": "差旅办法"})]
    cite_res = citation_injector.inject_citations("差旅每天享有 100 元交通费。", cite_c)
    assert "[1]" in cite_res.annotated_answer
    print("        ✓ Citation Injector passed.")

    # Test 12: Extractive Context Compressor
    print("[13/16] Testing Extractive Context Compressor...")
    long_chunk = ChunkItem(id="cx", content="公司创立于2010年。差旅每天补贴标准为100元。感谢大家遵守。")
    comp = context_compressor.compress_chunk("差旅每天补贴多少", long_chunk)
    assert "差旅每天补贴标准为100元" in comp.content
    print("        ✓ Context Compressor passed.")

    # Test 13: Query Rewriter
    print("[14/16] Testing Query Rewriter...")
    rw_text, rw_flag = query_rewriter.rewrite_query("它的上限是多少？", last_turn_query="差旅交通补贴规定")
    assert rw_flag is True
    print("        ✓ Query Rewriter passed.")

    # Test 14: Audit Sink & Dataset Export
    print("[15/16] Testing Audit Sink & Dataset Export...")
    await audit_sink.log_event("TEST_EVENT", {"k": "v"})
    exported = audit_sink.export_jev_finetuning_dataset()
    assert exported >= 0
    print("        ✓ Audit Sink passed.")

    # Test 15: Pipeline End-to-End with Citations & Radar
    print("[16/16] Testing End-to-End Pipeline with Domain Routing & Citations...")
    pipeline = RAGGatekeeperPipeline(engine)
    pipe_res = await pipeline.execute(
        PipelineRequest(
            query="内部政策：差旅发票住宿报销上限是多少？",
            candidate_chunks=[
                ChunkItem(id="c1", content="核心条款：差旅每天住宿补贴上限为450元。", metadata={"title": "差旅规程"}),
            ],
            generated_answer="差旅每天住宿补贴上限为450元。",
        )
    )
    assert pipe_res.short_circuited is False
    assert pipe_res.routed_domain == "FINANCE_EXPENSE"
    assert "[1]" in pipe_res.final_output
    print("        ✓ Full Pipeline End-to-End passed.")

    print("\nALL 16 VERIFICATION SUITES PASSED SUCCESSFULLY! (100% OK)")


if __name__ == "__main__":
    asyncio.run(main())
