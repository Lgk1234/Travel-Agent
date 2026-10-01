"""hotel_node：调用住宿检索能力获取住宿参考（不调用 LLM）。

关键词来自任务计划（默认取用户填写的住宿类型）。为贴近行程，优先**按景点群坐标就近检索**
（解决"远郊民宿配市中心景点"）；坐标取不到时回退城市检索。

耦合说明：坐标来自 route_node 已产出的文本，本节点用**自包含的容错正则**解析，
不 import route_node、不要求其输出格式稳定；解析不到即回退，不增加结构耦合。
"""
import logging
import re

from src.planning.state import AgentState, PlanningContext
from src.tools.amap.hotel import search_hotel, search_hotel_nearby

logger = logging.getLogger("planning.nodes.hotel")

# 容错匹配「经度,纬度」
_COORD_RE = re.compile(r"\d+\.\d+,\d+\.\d+")


def center_from_route(route_text: str) -> str:
    """从路线产出中解析景点坐标并取中心点；解析不到返回空串。"""
    coords = []
    for lonlat in _COORD_RE.findall(route_text or ""):
        try:
            lon, lat = lonlat.split(",")
            coords.append((float(lon), float(lat)))
        except ValueError:
            continue
    if not coords:
        return ""
    lon = sum(c[0] for c in coords) / len(coords)
    lat = sum(c[1] for c in coords) / len(coords)
    return f"{lon:.6f},{lat:.6f}"


async def hotel_node(state: AgentState, *, ctx: PlanningContext) -> dict:
    await ctx.emit("hotel", "正在检索住宿…")
    plan = state.get("plan") or {}
    keyword = (plan.get("hotel_keyword") or "").strip() or (ctx.request.accommodation or "酒店")

    collected = state.get("collected") or {}
    center = center_from_route(collected.get("route", ""))
    try:
        if center:
            text = await search_hotel_nearby(keyword, center, ctx.request.city)
        else:
            text = await search_hotel(keyword, ctx.request.city)
    except Exception as e:  # noqa: BLE001
        logger.warning("住宿检索失败：%s", e)
        text = f"（住宿检索失败：{e}）"
    return {"collected": {"hotel": text or "（未检索到住宿信息）"}}
