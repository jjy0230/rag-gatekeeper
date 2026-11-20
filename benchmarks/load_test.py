"""Benchmark and load testing suite for RAG-Gatekeeper.

Measures P50, P90, P99 latencies, short-circuit ratio, and token savings.
"""

import asyncio
import time
import statistics
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_gatekeeper.engine.mock_client import MockDecisionEngine
from rag_gatekeeper.pipeline import RAGGatekeeperPipeline
from rag_gatekeeper.models import PipelineRequest, ChunkItem


async def run_single_request(pipeline: RAGGatekeeperPipeline, query: str, chunks: list[ChunkItem]):
    start = time.perf_counter()
    res = await pipeline.execute(
        PipelineRequest(query=query, candidate_chunks=chunks)
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    return res, latency_ms


async def main():
    print("==================================================")
    print("Starting RAG-Gatekeeper Performance Benchmark")
    print("==================================================")

    # Simulated engine with realistic 6ms non-autoregressive Jev latency
    engine = MockDecisionEngine(simulated_latency_ms=6.0)
    pipeline = RAGGatekeeperPipeline(engine)

    queries = [
        # Common queries (expected to short-circuit)
        ("你好，快速排序的时间复杂度是多少？", False),
        ("Python 中字典怎么按值排序？", False),
        ("今天星期几？", False),
        ("1+1等于几？", False),
        # Enterprise domain queries (expected to retrieve)
        ("公司差旅报销的标准和上限是多少？", True),
        ("内部规范中关于年假折算的规定是什么？", True),
        ("SLA 服务等级协议中一级响应时间是多少？", True),
        ("Q3 财报公布的研发费用是多少？", True),
    ] * 25  # 200 total requests

    candidate_chunks = [
        ChunkItem(id=f"c_{i}", content=f"核心条款内容示例片段 {i}：相关政策与说明")
        for i in range(10)
    ]

    print(f"Executing {len(queries)} concurrent benchmark requests...")
    start_total = time.perf_counter()

    tasks = [
        run_single_request(pipeline, q[0], candidate_chunks)
        for q in queries
    ]
    results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_total
    latencies = [r[1] for r in results]
    responses = [r[0] for r in results]

    short_circuited_count = sum(1 for r in responses if r.short_circuited)
    retrieval_count = len(responses) - short_circuited_count

    p50 = statistics.median(latencies)
    latencies.sort()
    p90 = latencies[int(len(latencies) * 0.90)]
    p99 = latencies[int(len(latencies) * 0.99)]

    print("\n---------------- Benchmark Results ----------------")
    print(f"Total Requests:            {len(queries)}")
    print(f"Total Elapsed Time:        {total_time:.2f} s")
    print(f"Throughput (QPS):          {len(queries) / total_time:.1f} req/s")
    print(f"Stage 1 Short-circuit Rate:{short_circuited_count / len(queries) * 100:.1f}%")
    print(f"Vector DB Retrieval Saved: {short_circuited_count} calls")
    print(f"P50 Latency:               {p50:.2f} ms")
    print(f"P90 Latency:               {p90:.2f} ms")
    print(f"P99 Latency:               {p99:.2f} ms")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    asyncio.run(main())
