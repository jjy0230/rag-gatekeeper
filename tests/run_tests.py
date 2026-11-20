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
    print("Running RAG-Gatekeeper Full Verification")
    print("========================================")
    engine = MockDecisionEngine(simulated_latency_ms=0.0)

    # Test 0: Aho-Corasick Matcher
    print("[1/14] Testing Aho-Corasick Multi-Pattern Matcher...")
    matcher = AhoCorasickMatcher(["内部规范", "SLA", "差旅报销"])
    assert matcher.has_any("内部规范变更通知") is True
    assert matcher.has_any("公开常识问答") is False
    print("       ✓ Aho-Corasick Automaton passed.")

    # Test 1: Stage 1 Short Circuit
    print("[2/14] Testing Stage 1 Short-circuiting...")
    stage1 = RetrievalArbitrator(engine)
    res1 = await stage1.arbitrate(ArbitrationRequest(query="你好，请问快速排序的时间复杂度是多少？"))
    assert res1.needs_external_knowledge is False, f"Expected False, got {res1}"
    print("       ✓ Stage 1 Short-circuit passed.")

    # Test 2: Stage 1 Retrieval Trigger
    print("[3/14] Testing Stage 1 Knowledge Retrieval Trigger...")
    res2 = await stage1.arbitrate(ArbitrationRequest(query="请帮我查询我们部门最新的差旅报销审批标准"))
    assert res2.needs_external_knowledge is True, f"Expected True, got {res2}"
    print("       ✓ Stage 1 Trigger passed.")

    # Test 3: Stage 2 Rerank & Noise Filtering
    print("[4/14] Testing Stage 2 Reranking & Filtering...")
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
    print("[5/14] Testing Stage 2 Anti-Recall-Cliff Fallback...")
    low_chunks = [
        ChunkItem(id="c1", content="无关免责声明一"),
        ChunkItem(id="c2", content="无关免责声明二"),
    ]
    res4 = await stage2.rerank(RerankRequest(query="冷门问答", chunks=low_chunks))
    assert res4.fallback_applied is True, "Expected fallback_applied=True"
    assert res4.kept_count == 2, f"Expected 2 chunks retained in fallback, got {res4.kept_count}"
    print("       ✓ Stage 2 Fallback Safeguard passed.")

    # Test 5: Stage 3 Hallucination Block
    print("[6/14] Testing Stage 3 Hallucination Blocking...")
    stage3 = HallucinationGuardrail(engine)
    req5 = GuardrailRequest(
        query="公司允许带宠物上班吗？",
        answer="允许，并且提供猫粮补助。__MOCK_HALLUCINATION__",
        evidence_chunks=["员工守则第三条：严禁私自携带宠物进入办公区域。"],
    )
    res5 = await stage3.verify(req5)
    assert res5.blocked is True, "Expected hallucination to be blocked"
    assert res5.degraded is True
    assert res5.final_answer == settings.FALLBACK_TEMPLATE
    print("       ✓ Stage 3 Hallucination Block & Fallback Template passed.")

    # Test 6: Speculative Sentence Guardrail
    print("[7/14] Testing Speculative Sentence Guardrail...")
    s_guard = SpeculativeSentenceGuardrail(engine, early_kill_threshold=0.75)
    sentences, leftover = s_guard.split_sentences("第一句。第二句！未完")
    assert len(sentences) == 2
    is_h, p = await s_guard.verify_sentence("福利", "发一万。__MOCK_HALLUCINATION__", ["无此福利"])
    assert is_h is True
    print("       ✓ Speculative Sentence Guardrail passed.")

    # Test 7: Native In-Memory Hybrid Store
    print("[8/14] Testing Native Hybrid Vector Store (BM25 + Dense RRF)...")
    store = InMemoryHybridStore()
    store.upsert_chunks([
        ChunkItem(id="d1", content="差旅每天补贴 100 元"),
        ChunkItem(id="d2", content="年假制度说明"),
    ])
    matched = store.search("差旅补贴", top_k=1)
    assert len(matched) == 1 and matched[0].id == "d1"
    print("       ✓ Native Hybrid Store passed.")

    # Test 8: Semantic Cache L1 & L2
    print("[9/14] Testing Semantic Cache (L1 Exact + L2 Cosine)...")
    semantic_cache.clear()
    semantic_cache.set("年假未休完怎么处理？", "年假未休完可在年底折算补贴发放。")
    assert semantic_cache.get("年假未休完怎么处理？") is not None
    assert semantic_cache.get("如果年假没休完怎么处理呢") is not None
    print("       ✓ Semantic Cache passed.")

    # Test 9: Citation Footnotes Injector
    print("[10/14] Testing Citation & Footnote Injector...")
    cite_c = [ChunkItem(id="c1", content="差旅每天享有 100 元交通补贴。", metadata={"title": "差旅办法"})]
    cite_res = citation_injector.inject_citations("差旅每天享有 100 元交通费。", cite_c)
    assert "[1]" in cite_res.annotated_answer
    assert "参考来源：" in cite_res.annotated_answer
    print("        ✓ Citation Injector passed.")

    # Test 10: Extractive Context Compressor
    print("[11/14] Testing Extractive Context Compressor...")
    long_chunk = ChunkItem(id="cx", content="公司创立于2010年，环境优美。差旅每天补贴标准为100元。感谢大家遵守。")
    comp = context_compressor.compress_chunk("差旅每天补贴多少", long_chunk)
    assert "差旅每天补贴标准为100元" in comp.content
    print("        ✓ Context Compressor passed.")

    # Test 11: Query Rewriter
    print("[12/14] Testing Query Rewriter & Pronoun De-referencing...")
    rw_text, rw_flag = query_rewriter.rewrite_query("它的上限是多少？", last_turn_query="差旅交通补贴规定")
    assert rw_flag is True
    assert "差旅交通补贴规定" in rw_text
    print("        ✓ Query Rewriter passed.")

    # Test 12: Audit Sink & Dataset Export
    print("[13/14] Testing Audit Sink & Dataset Export...")
    await audit_sink.log_event("TEST_EVENT", {"k": "v"})
    exported = audit_sink.export_jev_finetuning_dataset()
    assert exported >= 0
    print("        ✓ Audit Sink passed.")

    # Test 13: Pipeline End-to-End
    print("[14/14] Testing End-to-End Pipeline with Citations...")
    pipeline = RAGGatekeeperPipeline(engine)
    pipe_res = await pipeline.execute(
        PipelineRequest(
            query="内部政策：年假没休完怎么办？",
            candidate_chunks=[
                ChunkItem(id="c1", content="核心条款：年假未休可折算补贴。", metadata={"title": "年假规程"}),
            ],
            generated_answer="未休完的年假可在年底按规定折算补贴。",
        )
    )
    assert pipe_res.short_circuited is False
    assert "[1]" in pipe_res.final_output
    print("        ✓ Pipeline End-to-End passed.")

    print("\nALL 14 VERIFICATION SUITES PASSED SUCCESSFULLY! (100% OK)")


if __name__ == "__main__":
    asyncio.run(main())
