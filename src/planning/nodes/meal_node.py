"""meal_node：调用高德 POI 能力检索**真实餐饮**（不调用 LLM）。

此前整条链路没有任何餐饮数据源，summary 只能按提示词的兜底要求把三餐写成
「XX周边餐厅 / XX小吃街」。本节点按任务计划给出的餐饮关键词检索真实餐厅 / 小吃 POI，
写入 collected["meal"]，供 summary_node 填写三餐时使用**真实店名**。
"""
import logging

from src.planning.state import AgentState, PlanningContext
from src.tools.amap import poi

logger = logging.getLogger("planning.nodes.meal")

# 餐饮关键词上限（控制工具调用次数；planner 动态生成 2~4 个）
MAX_MEAL_KEYWORDS = 4


def _default_keywords(city: str) -> list[str]:
    """任务计划没给餐饮关键词时的兜底，按品类分化以保留多样性。"""
    return [f"{city}特色早餐", f"{city}特色正餐", f"{city}特色小吃"]


async def meal_node(state: AgentState, *, ctx: PlanningContext) -> dict:
    await ctx.emit("meal", "正在检索当地餐饮…")
    plan = state.get("plan") or {}
    keywords = [k.strip() for k in (plan.get("meal_keywords") or []) if k and k.strip()]
    keywords = keywords[:MAX_MEAL_KEYWORDS] or _default_keywords(ctx.request.city)
    try:
        text = await poi.search_poi_many(keywords, ctx.request.city)
    except Exception as e:  # noqa: BLE001
        logger.warning("餐饮检索失败：%s", e)
        text = f"（餐饮检索失败：{e}）"
    return {"collected": {"meal": text or "（未检索到餐饮信息）"}}
