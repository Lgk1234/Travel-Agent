"""POI / 地点解析能力：检索场所、把地点名解析为经纬度。

复用 `client.py` 的唯一 MCP 连接，不新建连接。
"""
import logging
from typing import List

from src.tools.amap import client

# 复用 client 的 POI 精简格式化（同包内复用，避免把整段原始 JSON 灌进上下文）
_parse_poi = client._parse_poi

logger = logging.getLogger("tools.amap.poi")

# 单次行程规划允许的 POI 检索次数上限（防止工具调用爆炸拖慢整体流程）
MAX_SEARCHES = 6


async def _find_tool(name: str):
    tools = await client.get_amap_tools()
    if not tools:
        return None
    return next((t for t in tools if t.name == name), None)


async def search_poi(keywords: str, city: str = "") -> str:
    """按关键词检索 POI（确定性直查，不经 LLM）。"""
    return await client.amap_poi_search(keywords, city)


async def search_poi_many(keywords_list: List[str], city: str = "",
                          limit: int = MAX_SEARCHES) -> str:
    """按多个关键词检索并合并结果；关键词数量受上限约束，并发受 client 信号量保护。"""
    import asyncio

    kws = [k.strip() for k in (keywords_list or []) if k and k.strip()][:limit]
    if not kws:
        return ""
    results = await asyncio.gather(*(search_poi(k, city) for k in kws))
    blocks = [f"【{k}】\n{r}" for k, r in zip(kws, results) if r]
    return "\n\n".join(blocks)


async def geocode(address: str, city: str = "") -> str:
    """地点名/地址 → 经纬度（maps_geo），供路线计算使用。失败返回空串。"""
    tool = await _find_tool("maps_geo")
    if tool is None:
        return ""
    args = {"address": address}
    if city:
        args["city"] = city
    try:
        return await tool.ainvoke(args)
    except Exception as e:  # noqa: BLE001
        logger.warning("地理编码失败（%s）：%s", address, e)
        return ""


async def around_search(keywords: str, location: str, radius: str = "5000") -> str:
    """周边检索（maps_around_search）：以 location 为中心、radius 为半径搜索 POI。

    用于「按景点群就近找住宿」等场景。失败返回空串，由调用方回退。
    """
    tool = await _find_tool("maps_around_search")
    if tool is None:
        return ""
    try:
        raw = await tool.ainvoke(
            {"keywords": keywords, "location": location, "radius": radius}
        )
        # 压成「名称｜类型｜地址」精简列表，避免原始 JSON（含图片 URL）撑爆上下文
        return _parse_poi(raw if isinstance(raw, str) else str(raw))
    except Exception as e:  # noqa: BLE001
        logger.warning("周边检索失败（%s @ %s）：%s", keywords, location, e)
        return ""
