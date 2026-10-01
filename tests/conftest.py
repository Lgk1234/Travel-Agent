"""pytest 公共 fixture：屏蔽外部依赖（高德 MCP / 和风 / 真实 LLM），保持离线可跑。"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# 保证 `from src...` 可导入
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning.state import TaskPlan  # noqa: E402
from src.schemas.trip import Budget, DayPlan, TripPlan  # noqa: E402


def _mock_trip_plan() -> TripPlan:
    return TripPlan(
        city="北京",
        start_date="2026-10-01",
        end_date="2026-10-03",
        days=[DayPlan(date="2026-10-01", day_index=0, description="mock")],
        weather_info=[],
        overall_suggestions="mock",
        budget=Budget(total_attractions=200, total_hotels=800,
                      total_meals=600, total_transportation=400, total=2000),
    )


class _BoundLLM:
    """绑定了具体 schema 的假结构化输出模型。"""

    def __init__(self, schema):
        self._schema = schema

    async def ainvoke(self, messages=None, config=None):
        if getattr(self._schema, "__name__", "") == "TaskPlan":
            return TaskPlan(
                attraction_keywords=["北京 博物馆", "北京 胡同"],
                hotel_keyword="经济型酒店",
                route_requirements="优先地铁",
                reason="mock",
            )
        return _mock_trip_plan()


class FakeLLM:
    """离线替代真实 ChatOpenAI：with_structured_output 返回独立绑定实例。"""

    def with_structured_output(self, schema):
        return _BoundLLM(schema)


@pytest.fixture(autouse=True)
def _patch_external(monkeypatch):
    import src.planning.graph as graph_mod
    import src.planning.nodes.attraction_node as an
    import src.planning.nodes.hotel_node as hn
    import src.planning.nodes.meal_node as mealn
    import src.planning.nodes.route_node as rn
    import src.planning.nodes.weather_node as wn

    # ---- 入口：模型与 MCP 工具 ----
    async def fake_get_tools():
        return []

    monkeypatch.setattr(graph_mod, "get_amap_tools", fake_get_tools)
    monkeypatch.setattr(graph_mod, "get_llm", lambda *a, **k: FakeLLM())

    # ---- weather_node：和风天气工具 ----
    async def fake_weather_ainvoke(*a, **k):
        return "【北京 天气预报】（来源：和风天气）\n- 2026-10-01：晴 8~22℃"

    monkeypatch.setattr(wn, "weather_tool",
                        MagicMock(ainvoke=AsyncMock(side_effect=fake_weather_ainvoke)))

    # ---- attraction_node：高德 POI 能力 ----
    async def fake_search_poi_many(keywords, city=""):
        return ("- 故宫博物院｜博物馆｜景山前街4号\n"
                "- 天安门广场｜景点｜长安街\n"
                "- 南锣鼓巷｜街区｜南锣鼓巷")

    monkeypatch.setattr(an, "poi", MagicMock(search_poi_many=AsyncMock(side_effect=fake_search_poi_many)))

    # ---- hotel_node：住宿检索能力 ----
    async def fake_search_hotel(keyword, city=""):
        return "（酒店参考）如家/汉庭等连锁，价格以平台为准"

    async def fake_search_hotel_nearby(keyword, location, city="", radius="5000"):
        return "（酒店参考·就近）如家/汉庭等连锁，价格以平台为准"

    monkeypatch.setattr(hn, "search_hotel", fake_search_hotel)
    monkeypatch.setattr(hn, "search_hotel_nearby", fake_search_hotel_nearby)

    # ---- meal_node：餐饮检索能力（提供真实店名，避免「周边餐厅」兜底）----
    async def fake_search_meal(keywords, city=""):
        return "- 全聚德烤鸭店｜餐饮｜前门大街30号\n- 姚记炒肝｜小吃｜鼓楼东大街311号"

    monkeypatch.setattr(mealn, "poi", MagicMock(search_poi_many=AsyncMock(side_effect=fake_search_meal)))

    # ---- route_node：地理编码 + 路线能力 ----
    async def fake_geocode(address, city=""):
        return f"{address}｜116.397,39.908｜兴趣点"

    async def fake_transit(origin, destination, city="", cityd=""):
        return "起点：x 终点：y\n方案1｜约30分钟"

    monkeypatch.setattr(rn, "poi", MagicMock(geocode=AsyncMock(side_effect=fake_geocode)))
    monkeypatch.setattr(rn, "route", MagicMock(transit=AsyncMock(side_effect=fake_transit)))
