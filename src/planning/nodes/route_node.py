"""route_node：调用高德路线能力计算景点间接驳（不调用 LLM）。

依赖 attraction_node 的产出：先把景点名解析为经纬度（POI 地理编码），
再对相邻景点计算公共交通接驳，全程受次数上限约束。
"""
import asyncio
import logging
import re

from src.planning.state import AgentState, PlanningContext
from src.tools.amap import poi, route

logger = logging.getLogger("planning.nodes.route")

# 参与接驳计算的景点数上限（含坐标解析与路线查询，避免工具调用爆炸）
MAX_POINTS = 6

_COORD_RE = re.compile(r"^\d+\.?\d*,\d+\.?\d*$")


def _names_from_poi(text: str) -> list[str]:
    """按【分组】取每组**首个（主）景点名**。

    POI 检索结果按关键词分组，每组首条是主景点，其余多为检票处 / 某某门 / 停车场等子点。
    若逐行取前 N 条，会拿到一堆同一景区的子点（曾出现 6 个"景点"全是故宫的
    检票处 / 午门 / 西南角楼，导致接驳计算毫无意义）。
    """
    lines = [line.strip() for line in (text or "").splitlines()]
    # 无【分组】标题时退回「取全部条目」，避免只拿到 1 个名字
    has_groups = any(line.startswith("【") for line in lines)

    names: list[str] = []
    group_taken = False
    for line in lines:
        if line.startswith("【"):
            group_taken = False  # 进入新分组
            continue
        if not line.startswith("- "):
            continue
        if has_groups and group_taken:
            continue  # 该组已取主景点，跳过子点
        name = line[2:].split("｜")[0].strip()
        if name and name not in names:
            names.append(name)
        if has_groups:
            group_taken = True
    return names


def _first_location(text: str) -> str:
    """从地理编码精简文本（`地址｜经度,纬度｜级别`）中取首个坐标。"""
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split("｜")]
        if len(parts) >= 2 and _COORD_RE.match(parts[1]):
            return parts[1]
    return ""


async def route_node(state: AgentState, *, ctx: PlanningContext) -> dict:
    await ctx.emit("routing", "正在规划景点间接驳…")
    city = ctx.request.city
    collected = state.get("collected") or {}
    names = _names_from_poi(collected.get("attraction", ""))[:MAX_POINTS]
    if len(names) < 2:
        return {"collected": {"route": "（可规划的景点不足，跳过接驳计算）"}}

    # 1) 景点名 → 经纬度（并发，受 client 信号量限流保护）
    geos = await asyncio.gather(*(poi.geocode(n, city) for n in names))
    coords: list[tuple[str, str]] = []
    for n, g in zip(names, geos):
        loc = _first_location(g)
        if loc:
            coords.append((n, loc))

    if len(coords) < 2:
        return {"collected": {"route": "（景点坐标解析失败，无法计算接驳）"}}

    # 2) 相邻景点接驳
    lines = ["景点坐标："] + [f"- {n}｜{loc}" for n, loc in coords]
    lines.append("相邻接驳：")
    for (n1, c1), (n2, c2) in zip(coords, coords[1:]):
        r = await route.transit(c1, c2, city, city)
        lines.append(f"- {n1} → {n2}：{r}")

    return {"collected": {"route": "\n".join(lines)}}
