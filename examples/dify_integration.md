# Dify 与 RAG-Gatekeeper 零代码集成配置指南

RAG-Gatekeeper 提供了完全兼容 OpenAI API 规范的 `/v1/chat/completions` 反向代理端点，可直接作为 Dify 的自定义模型供应商（Custom Model Provider）接入。

---

## 一、 Dify 接入配置步骤

1. 登录 Dify 管理后台，点击右上角头像 ➔ **设置（Settings）** ➔ **模型供应商（Model Providers）**；
2. 找到 **OpenAI-API-compatible（兼容 OpenAI API 的模型）**，点击 **添加模型（Add Model）**；
3. 填写连接配置：
   * **模型类型**：`LLM`
   * **模型名称**：`rag-gatekeeper`（或自定义名称）
   * **API Key**：`none`（可填任意非空字符串）
   * **API Base URL**：`http://<your-gatekeeper-host>:8080/v1`
4. 点击 **保存（Save）**。

---

## 二、 业务效果与时序特性

1. **常识与闲聊问题**：
   * 自动在阶段一短路跳过后续知识库检索，直接返回快速解答，显著降低向量检索计算负载。
2. **知识库问答场景**：
   * 支持通过 SSE 流式正常打字返回；
   * 在响应头或流式末尾输出 Gatekeeper 审计元数据（短路标记、决策耗时与置信度）。
