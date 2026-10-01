"""Chat Agent（问答页入口，Agent 层）。

用户提问 → LLM 判断需求 → **直接调用 tools** 取数 → SSE 流式返回。
本入口不使用 Planner / Worker，也不做结构化行程规划（那是 Planning Agent 的职责）。

工具来源（统一能力层）：
- 高德 MCP 工具列表（POI / 路线 / 地理编码 / 天气等地图能力）
- 和风天气工具（tools/weather_tool，天气由和风提供）

短期记忆：LangGraph MemorySaver checkpointer + thread_id 实现多轮上下文。
"""
import logging
from typing import AsyncGenerator, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from src.llm import get_llm
from src.observability import LocalTraceCollector, run_config
from src.prompts.prompts import CHAT_AGENT_PROMPT
from src.schemas.chat import ChatRequest
from src.tools.amap.client import get_amap_tools
from src.tools.weather_tool import weather_tool

logger = logging.getLogger("chat_agent")


class ChatAgent:
    """问答 Agent：单 ReAct 智能体 + 工具直调 + 短期记忆 + 流式输出。"""

    def __init__(self):
        self.tools = None
        self._agents: dict = {}
        # 短期记忆：跨多轮对话的 checkpoint（按 thread_id 区分会话）
        self._checkpointer = MemorySaver()

    async def _ensure_tools(self):
        if self.tools is None:
            amap_tools = await get_amap_tools()
            # 天气走和风，地图能力走高德 MCP；统一挂载，由 LLM 自主判断调用哪个
            self.tools = [*amap_tools, weather_tool]
            logger.info("Chat Agent 工具加载完成（高德 %d 个 + 和风天气 1 个）", len(amap_tools))

    def _get_agent(self, model: Optional[str]):
        key = model or "default"
        if key not in self._agents:
            self._agents[key] = create_agent(
                get_llm(model, disable_thinking=True),
                self.tools,
                system_prompt=CHAT_AGENT_PROMPT,
                checkpointer=self._checkpointer,
            )
            logger.info("Chat Agent 已创建（model=%s，已启用短期记忆）", key)
        return self._agents[key]

    @staticmethod
    def _build_messages(request: ChatRequest):
        messages = [SystemMessage(content=CHAT_AGENT_PROMPT)]
        for m in request.history:
            if m.role == "user":
                messages.append(HumanMessage(content=m.content))
            else:
                messages.append(AIMessage(content=m.content))
        messages.append(HumanMessage(content=request.message))
        return messages

    async def astream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """流式产出助手回复文本片段。"""
        await self._ensure_tools()
        agent = self._get_agent(request.model)

        # 短期记忆：有 thread_id 则依赖 checkpoint（只发最新消息）；
        # 否则手动拼历史（向后兼容无记忆的旧调用方式）。
        if request.thread_id:
            messages = [HumanMessage(content=request.message)]
            cfg: dict = {"configurable": {"thread_id": request.thread_id}}
        else:
            messages = self._build_messages(request)
            cfg = {}

        # 本地 trace 落盘：必须在 run_config 之前 activate，
        # 这样 run_config 才会把 handler 注入 callbacks。
        local = LocalTraceCollector("chat")
        local.activate()
        cfg.update(run_config("chat"))

        try:
            async for event in agent.astream_events({"messages": messages}, cfg, version="v2"):
                if event.get("event") == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    content = getattr(chunk, "content", None)
                    if isinstance(content, str) and content:
                        yield content
        finally:
            local.deactivate()
            local.dump()
