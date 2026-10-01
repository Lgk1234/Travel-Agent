"""路线 / 距离能力：接驳方式、时长、距离。

复用 `client.py` 的唯一 MCP 连接，不新建连接。
坐标格式统一为「经度,纬度」（与高德 MCP 入参一致）。
"""
import logging

from src.tools.amap import client

logger = logging.getLogger("tools.amap.route")

# 单次行程规划允许的路线查询次数上限
MAX_ROUTE_CALLS = 8


async def _find_tool(name: str):
    tools = await client.get_amap_tools()
    if not tools:
        return None
    return next((t for t in tools if t.name == name), None)


async def _invoke(tool_name: str, args: dict, label: str) -> str:
    """调用指定路线工具；失败返回明确降级文本，绝不编造时长/距离。"""
    tool = await _find_tool(tool_name)
    if tool is None:
        return f"（{label}工具不可用）"
    try:
        return await tool.ainvoke(args)
    except Exception as e:  # noqa: BLE001
        logger.warning("%s 查询失败：%s", label, e)
        return f"（{label}查询失败：{e}）"


async def transit(origin: str, destination: str, city: str = "", cityd: str = "") -> str:
    """公交/地铁接驳（maps_direction_transit_integrated）。"""
    args = {"origin": origin, "destination": destination}
    if city:
        args["city"] = city
    if cityd:
        args["cityd"] = cityd
    return await _invoke("maps_direction_transit_integrated", args, "公交路线")


async def driving(origin: str, destination: str) -> str:
    """驾车接驳（maps_direction_driving）。"""
    return await _invoke("maps_direction_driving",
                         {"origin": origin, "destination": destination}, "驾车路线")


async def walking(origin: str, destination: str) -> str:
    """步行接驳（maps_direction_walking）。"""
    return await _invoke("maps_direction_walking",
                         {"origin": origin, "destination": destination}, "步行路线")


async def distance(origins: str, destination: str, type_: str = "1") -> str:
    """批量距离测量（maps_distance）：origins 支持竖线分隔多个起点。

    type_: 1-驾车距离，0-直线距离，3-步行距离。
    """
    return await _invoke("maps_distance",
                         {"origins": origins, "destination": destination, "type": type_},
                         "距离测量")
