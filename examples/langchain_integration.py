"""Example: Zero-code integration with LangChain using OpenAI-compatible proxy.

Shows how to connect LangChain's ChatOpenAI directly through RAG-Gatekeeper.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Simulated LangChain-style usage:
# When using langchain_openai:
#
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage
#
# llm = ChatOpenAI(
#     base_url="http://localhost:8080/v1",
#     api_key="none",
#     model="qwen-2.5-7b-instruct",
#     streaming=True,
# )
#
# response = llm.invoke([HumanMessage(content="你好，请问快速排序的时间复杂度？")])
# print(response.content)

import asyncio
import httpx


async def test_langchain_proxy_call():
    print("Simulating LangChain ChatOpenAI call to RAG-Gatekeeper at http://localhost:8080/v1...")
    
    # 1. Test short-circuited conversational query
    payload_short = {
        "model": "qwen2.5-7b",
        "messages": [{"role": "user", "content": "你好，请问快速排序的时间复杂度是多少？"}],
        "stream": False,
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post("http://localhost:8080/v1/chat/completions", json=payload_short)
            print("Status Code:", resp.status_code)
            print("Headers:", resp.headers.get("X-Gatekeeper-Short-Circuited"))
            print("Response:", resp.json()["choices"][0]["message"]["content"])
        except httpx.ConnectError:
            print("Gateway is not currently running locally. Start it with:")
            print("uvicorn rag_gatekeeper.server:app --port 8080")


if __name__ == "__main__":
    asyncio.run(test_langchain_proxy_call())
