"""OpenAI-compatible reverse proxy handler supporting both streaming (SSE) and non-streaming modes."""

import json
import time
import logging
from typing import AsyncGenerator, Optional
import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from rag_gatekeeper.config import settings
from rag_gatekeeper.stages.stage1_arbitration import RetrievalArbitrator
from rag_gatekeeper.stages.stage2_rerank import ChunkReranker
from rag_gatekeeper.stages.stage3_guardrail import HallucinationGuardrail
from rag_gatekeeper.models import (
    ArbitrationRequest,
    GuardrailRequest,
    ChunkItem,
    RerankRequest,
)

logger = logging.getLogger(__name__)


class GatewayProxy:
    """Manages OpenAI-compatible proxying, SSE streaming interception, and guardrail verification."""

    def __init__(
        self,
        stage1: RetrievalArbitrator,
        stage2: ChunkReranker,
        stage3: HallucinationGuardrail,
    ):
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self._http_client: Optional[httpx.AsyncClient] = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
            self._http_client = httpx.AsyncClient(
                timeout=settings.UPSTREAM_TIMEOUT_SEC, limits=limits
            )
        return self._http_client

    def extract_user_query(self, messages: list[dict]) -> str:
        """Extract the last user prompt from OpenAI messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                elif isinstance(content, list):
                    # Multi-modal payload: extract text blocks
                    texts = [b.get("text", "") for b in content if b.get("type") == "text"]
                    return " ".join(texts).strip()
        return ""

    async def handle_chat_completion(self, body: dict, request: Request) -> Response:
        """Main dispatcher for /v1/chat/completions."""
        messages = body.get("messages", [])
        is_stream = body.get("stream", False)
        user_query = self.extract_user_query(messages)

        # ----------------------------------------------------
        # Step 1: Stage 1 Retrieval Arbitration Gate (~8ms)
        # ----------------------------------------------------
        arb_res = await self.stage1.arbitrate(ArbitrationRequest(query=user_query))

        headers = {
            "X-Gatekeeper-Short-Circuited": str(not arb_res.needs_external_knowledge).lower(),
            "X-Gatekeeper-Arbitration-Latency": f"{arb_res.latency_ms:.2f}ms",
            "X-Gatekeeper-Arbitration-Confidence": f"{arb_res.confidence:.2f}",
        }

        # If upstream LLM URL is not configured, run in mock server mode
        if not settings.UPSTREAM_LLM_URL:
            if is_stream:
                return StreamingResponse(
                    self._mock_stream_generator(user_query, arb_res),
                    media_type="text/event-stream",
                    headers=headers,
                )
            else:
                return self._mock_non_stream_response(user_query, arb_res, headers)

        # Upstream proxying mode
        if is_stream:
            return StreamingResponse(
                self._upstream_stream_proxy(body, user_query, arb_res),
                media_type="text/event-stream",
                headers=headers,
            )
        else:
            return await self._upstream_non_stream_proxy(body, user_query, arb_res, headers)

    def _mock_non_stream_response(self, user_query: str, arb_res, headers: dict) -> Response:
        """Return simulated OpenAI chat completion when upstream is not configured."""
        status_text = (
            "【检索已短路】此提问判定为通用常识或无需检索，已绕过向量数据库。"
            if not arb_res.needs_external_knowledge
            else "【放行检索】此提问需要企业私有知识库支持，已调度向量检索。"
        )
        content = (
            f"{status_text}\n\n"
            f"问题: {user_query}\n"
            f"判定置信度: {arb_res.confidence:.2f} (耗时: {arb_res.latency_ms:.2f}ms)"
        )
        resp_payload = {
            "id": f"chatcmpl-gk-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "rag-gatekeeper-proxy",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 30, "total_tokens": 45},
            "gatekeeper": {
                "short_circuited": not arb_res.needs_external_knowledge,
                "arbitration_latency_ms": arb_res.latency_ms,
            },
        }
        from starlette.responses import JSONResponse
        return JSONResponse(content=resp_payload, headers=headers)

    async def _mock_stream_generator(self, user_query: str, arb_res) -> AsyncGenerator[str, None]:
        """Generate mock OpenAI SSE tokens followed by Gatekeeper telemetry trailer."""
        chunks = [
            "【Gatekeeper 路由】",
            " 判定结果: " + ("跳过检索" if not arb_res.needs_external_knowledge else "执行向量检索"),
            f" (耗时 {arb_res.latency_ms:.2f}ms)。",
            "\n这是模拟下游大模型的流式打字输出内容。",
        ]
        chat_id = f"chatcmpl-gk-{int(time.time())}"
        for part in chunks:
            chunk_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "rag-gatekeeper-proxy",
                "choices": [
                    {"index": 0, "delta": {"content": part}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        # Final stop chunk
        stop_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "rag-gatekeeper-proxy",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"

        # Gatekeeper audit trailer event
        trailer = {
            "type": "gatekeeper_audit",
            "short_circuited": not arb_res.needs_external_knowledge,
            "arbitration_latency_ms": arb_res.latency_ms,
            "stage3_checked": False,
        }
        yield f"data: {json.dumps(trailer, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    async def _upstream_non_stream_proxy(
        self, body: dict, user_query: str, arb_res, headers: dict
    ) -> Response:
        """Relay non-streaming request to upstream LLM with Stage 3 verification."""
        client = self._get_http_client()
        upstream_url = settings.UPSTREAM_LLM_URL
        req_headers = {"Authorization": f"Bearer {settings.UPSTREAM_LLM_API_KEY}"}

        upstream_resp = await client.post(upstream_url, json=body, headers=req_headers)
        data = upstream_resp.json()

        # Extract generated answer
        answer_text = ""
        if "choices" in data and data["choices"]:
            answer_text = data["choices"][0].get("message", {}).get("content", "")

        # Stage 3 Factuality Guardrail
        evidence_chunks = body.get("evidence_chunks", [])
        if evidence_chunks and answer_text:
            guard_res = await self.stage3.verify(
                GuardrailRequest(
                    query=user_query,
                    answer=answer_text,
                    evidence_chunks=evidence_chunks,
                )
            )
            headers["X-Gatekeeper-Guardrail-Blocked"] = str(guard_res.blocked).lower()
            headers["X-Gatekeeper-Hallucination-Prob"] = f"{guard_res.hallucination_prob:.2f}"
            if guard_res.blocked:
                # Replace answer content with degraded template
                data["choices"][0]["message"]["content"] = guard_res.final_answer
                data["gatekeeper_guardrail"] = {
                    "blocked": True,
                    "degraded": True,
                    "prob": guard_res.hallucination_prob,
                }

        from starlette.responses import JSONResponse
        return JSONResponse(content=data, status_code=upstream_resp.status_code, headers=headers)

    async def _upstream_stream_proxy(
        self, body: dict, user_query: str, arb_res
    ) -> AsyncGenerator[str, None]:
        """Relay SSE streaming chunks from upstream LLM and append audit event."""
        client = self._get_http_client()
        upstream_url = settings.UPSTREAM_LLM_URL
        req_headers = {"Authorization": f"Bearer {settings.UPSTREAM_LLM_API_KEY}"}

        collected_answer: list[str] = []
        async with client.stream("POST", upstream_url, json=body, headers=req_headers) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    payload_str = line[6:].strip()
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(payload_str)
                        delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            collected_answer.append(delta["content"])
                    except Exception:
                        pass
                yield f"{line}\n\n"

        # Verify completed answer asynchronously against evidence
        evidence_chunks = body.get("evidence_chunks", [])
        full_answer = "".join(collected_answer)
        audit_payload = {
            "type": "gatekeeper_audit",
            "short_circuited": not arb_res.needs_external_knowledge,
            "hallucination_detected": False,
        }
        if evidence_chunks and full_answer:
            guard_res = await self.stage3.verify(
                GuardrailRequest(
                    query=user_query,
                    answer=full_answer,
                    evidence_chunks=evidence_chunks,
                )
            )
            audit_payload["hallucination_detected"] = guard_res.blocked
            audit_payload["hallucination_prob"] = guard_res.hallucination_prob

        yield f"data: {json.dumps(audit_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
