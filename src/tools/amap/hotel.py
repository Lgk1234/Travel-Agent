"""住宿检索能力（基于 POI 检索 + 降级规则）。

复用 `client.py` 的唯一 MCP 连接，不新建连接。
"""
import logging

from src.tools.amap import poi

logger = logging.getLogger("tools.amap.hotel")


async def search_hotel(keyword: str, city: str = "") -> str:
    """按住宿偏好检索；高德对「民宿/客栈」等覆盖较弱时自动回退到「酒店」。

    只返回真实 POI 信息，不虚构价格。
    """
    keyword = (keyword or "").strip() or "酒店"
    result = await poi.search_poi(keyword, city)
    if "未检索到" in result and keyword != "酒店":
        logger.info("住宿关键词「%s」未命中，回退检索「酒店」", keyword)
        result = await poi.search_poi("酒店", city)
    return result


async def search_hotel_nearby(keyword: str, location: str, city: str = "",
                              radius: str = "5000") -> str:
    """按景点坐标就近检索住宿（解决"远郊住宿配市中心景点"）。

    先以 location 为中心做周边检索；周边无结果或解析失败时回退城市检索。
    """
    keyword = (keyword or "").strip() or "酒店"
    if location:
        result = await poi.around_search(keyword, location, radius)
        if result and "未检索到" not in result:
            return result
        logger.info("周边住宿未命中（%s @ %s），回退城市检索", keyword, location)
    return await search_hotel(keyword, city)
