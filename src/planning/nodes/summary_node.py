"""summary_node：LLM 调用点 2 —— 汇总工具结果，生成结构化 TripPlan。

输入是四个取数 node 采集到的真实数据；不编造数值，数据缺失时如实标注。
"""
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.observability import run_config
from src.planning.state import AgentState, PlanningContext
from src.prompts.prompts import PLANNER_AGENT_PROMPT
from src.schemas.trip import TripPlan

logger = logging.getLogger("planning.nodes.summary")

# 每个 section 的最大字符数，控制汇总上下文规模
_SECTION_MAX = 2000


def _trim(text: str, n: int = _SECTION_MAX) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "\n…（已截断）"


async def summary_node(state: AgentState, *, ctx: PlanningContext) -> dict:
    await ctx.emit("synthesize", "正在生成你的专属行程…")
    collected = state.get("collected") or {}
    sections = []
    for key, label in (
        ("weather", "天气信息"),
        ("attraction", "景点信息"),
        ("route", "路线接驳信息"),
        ("hotel", "酒店信息"),
        ("meal", "餐饮信息"),
        ("budget", "预算参考"),
    ):
        val = collected.get(key)
        if val:
            sections.append(f"【{label}】\n{_trim(val)}")

    synthesis = (
        f"用户需求：\n{ctx.req_text}\n\n"
        + "\n\n".join(sections)
        + "\n\n请严格按系统提示词的结构生成完整行程计划。"
    )
    plan: TripPlan = await ctx.summary_llm.ainvoke(
        [SystemMessage(content=PLANNER_AGENT_PROMPT), HumanMessage(content=synthesis)],
        config=run_config("summary"),
    )
    return {"collected": {"trip_plan": plan}}
