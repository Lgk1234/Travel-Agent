"""attraction_node：调用高德 POI 能力检索景点（不调用 LLM）。

检索关键词来自 planner_node 产出的任务计划，数量受上限约束。
"""
import logging

from src.planning.state import AgentState, PlanningContext
from src.tools.amap import poi

logger = logging.getLogger("planning.nodes.attraction")


async def attraction_node(state: AgentState, *, ctx: PlanningContext) -> dict:
    await ctx.emit("attraction", "正在检索景点…")
    plan = state.get("plan") or {}
    kws = plan.get("attraction_keywords") or []
    if not kws:  # 计划缺失时的兜底
        kws = [f"{ctx.request.city}热门景点"]
    try:
        text = await poi.search_poi_many(kws, ctx.request.city)
    except Exception as e:  # noqa: BLE001
        logger.warning("景点检索失败：%s", e)
        text = f"（景点检索失败：{e}）"
    return {"collected": {"attraction": text or "（未检索到景点信息）"}}
