"""Standalone Mock vLLM-Jev server simulating mode-io/vllm-jev /v1/systemone API.

Can be run locally to simulate true distributed network conditions without GPU deployment.
"""

import asyncio
import time
from typing import Any, Optional
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Mock vLLM-Jev Serving Engine")


class SystemOneRequest(BaseModel):
    model: str
    primitive: str
    context: str
    question: str
    system_instruction: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


@app.get("/health")
async def health():
    return {"status": "ready", "engine": "mode-io/vllm-jev-simulated"}


@app.post("/v1/systemone")
async def handle_system_one(req: SystemOneRequest):
    # Simulate non-autoregressive fast classification latency (~8ms)
    await asyncio.sleep(0.008)

    ctx_lower = req.context.lower()
    q_lower = req.question.lower()

    if req.primitive == "noul":
        if "needs_external_knowledge" in q_lower:
            common = ["你好", "hello", "1+1", "快速排序", "python"]
            if any(c in ctx_lower for c in common) and not any(k in ctx_lower for k in ["内部", "公司", "制度", "规范"]):
                return {"value": False, "prob_true": 0.05, "confidence": 0.95}
            return {"value": True, "prob_true": 0.92, "confidence": 0.92}

        if "contains_unsupported_facts" in q_lower:
            if "虚构" in ctx_lower or "__MOCK_HALLUCINATION__" in ctx_lower:
                return {"value": True, "prob_true": 0.88, "confidence": 0.88}
            return {"value": False, "prob_true": 0.06, "confidence": 0.94}

        return {"value": False, "prob_true": 0.5, "confidence": 0.5}

    elif req.primitive == "score":
        if "无关" in ctx_lower or "免责声明" in ctx_lower:
            score = 1.6
        elif "核心条款" in ctx_lower or "标准解答" in ctx_lower:
            score = 4.8
        else:
            score = 4.1
        return {
            "score": score,
            "distribution": {1: 0.05, 2: 0.05, 3: 0.1, 4: 0.4, 5: 0.4},
        }

    elif req.primitive == "choice":
        options = (req.parameters or {}).get("options", ["option_a", "option_b"])
        return {
            "selected": options[0] if options else "none",
            "probabilities": {opt: 1.0 / len(options) for opt in options},
        }

    return {"error": "unsupported primitive"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
