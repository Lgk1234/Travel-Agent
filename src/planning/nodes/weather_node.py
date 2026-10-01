"""weather_node：调用天气工具获取真实天气（不调用 LLM）。

天气走和风天气（tools/weather_tool：GeoAPI 定位 → Weather API 查询），
和风不可用时会自行回退高德天气；两者都失败返回明确降级文本，不编造数值。
"""
import logging

from src.planning.state import AgentState, PlanningContext
from src.tools.weather_tool import weather_tool

logger = logging.getLogger("planning.nodes.weather")


async def weather_node(state: AgentState, *, ctx: PlanningContext) -> dict:
    await ctx.emit("weather", "正在查询目的地天气…")
    req = ctx.request
    try:
        # 传入行程起止日期：逐日预报按该区间筛选，避免漏掉行程后几天
        text = await weather_tool.ainvoke({
            "city": req.city,
            "days": req.travel_days,
            "start_date": req.start_date,
            "end_date": req.end_date,
        })
    except Exception as e:  # noqa: BLE001
        logger.warning("天气 node 失败：%s", e)
        text = f"（天气查询失败：{e}；已降级）"
    return {"collected": {"weather": text or "（天气数据暂不可用）"}}
