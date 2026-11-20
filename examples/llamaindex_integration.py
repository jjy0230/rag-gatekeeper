"""Example: Connecting LlamaIndex to RAG-Gatekeeper.

Demonstrates hooking Stage 1 arbitration and Stage 2 reranker as custom LlamaIndex node postprocessors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from rag_gatekeeper.engine import get_engine
from rag_gatekeeper.stages.stage1_arbitration import RetrievalArbitrator
from rag_gatekeeper.stages.stage2_rerank import ChunkReranker
from rag_gatekeeper.models import ArbitrationRequest, RerankRequest, ChunkItem


async def run_llamaindex_demo():
    print("==================================================")
    print("LlamaIndex Pipeline Integration with RAG-Gatekeeper")
    print("==================================================")

    engine = get_engine()
    stage1 = RetrievalArbitrator(engine)
    stage2 = ChunkReranker(engine)

    query = "公司差旅交通补贴一天上限多少？"

    # Step 1: Pre-retrieval routing gate
    print(f"\n[Step 1] Evaluating Query Routing: '{query}'")
    arb_res = await stage1.arbitrate(ArbitrationRequest(query=query))
    print(f"         Requires Retrieval: {arb_res.needs_external_knowledge}")
    print(f"         Decision Latency:   {arb_res.latency_ms:.2f} ms")

    if not arb_res.needs_external_knowledge:
        print("         -> Short-circuiting vector index search!")
        return

    # Step 2: Post-retrieval reranking gate (filtering nodes)
    print("\n[Step 2] Filtering LlamaIndex Retrieved Nodes...")
    raw_nodes = [
        ChunkItem(id="node_1", content="核心条款：差旅每天交通补贴为 100 元，凭发票报销。"),
        ChunkItem(id="node_2", content="免责声明与无关广告推广文本。"),
        ChunkItem(id="node_3", content="公司年假制度补充说明。"),
    ]
    rerank_res = await stage2.rerank(RerankRequest(query=query, chunks=raw_nodes))
    print(f"         Nodes Evaluated: {rerank_res.total_evaluated}")
    print(f"         Nodes Retained:  {rerank_res.kept_count}")
    for item in rerank_res.filtered_chunks:
        print(f"         - [Score {item.score:.1f}] {item.chunk.content}")

    print("\nLlamaIndex Integration Pipeline completed successfully.")


if __name__ == "__main__":
    asyncio.run(run_llamaindex_demo())
