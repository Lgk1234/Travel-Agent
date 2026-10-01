"""集成测试：外部依赖全 mock 下跑通规划编排。

链路：planner_node(LLM) → weather → attraction → route → hotel → summary_node(LLM)
取数 node 不调用 LLM，全链路仅 2 次 LLM（planner + summary）。
不使用 pytest-asyncio，改为在同步用例内 asyncio.run，降低额外依赖。
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning.graph import TripPlanner
from src.schemas.trip import TripPlan, TripRequest


def _request() -> TripRequest:
    return TripRequest(
        city="北京",
        start_date="2026-10-01",
        end_date="2026-10-03",
        travel_days=3,
        people_count=2,
        transportation="地铁",
        accommodation="经济型酒店",
    )


def test_plan_trip_orchestration():
    planner = TripPlanner()
    events = []

    async def on_progress(stage, detail):
        events.append(stage)

    plan = asyncio.run(planner.plan_trip(_request(), on_progress=on_progress))

    assert isinstance(plan, TripPlan)
    assert plan.city == "北京"
    # 预算必须真实可用（禁止全 0）
    assert plan.budget is not None and plan.budget.total > 0
    # 节点顺序与进度事件（含新增的 budget，且 budget 在 synthesize 之前）
    for stage in ("planning", "weather", "attraction", "routing", "hotel", "meal",
                  "budget", "synthesize", "done"):
        assert stage in events
    assert events.index("meal") < events.index("budget") < events.index("synthesize")


def test_revise_trip_returns_plan():
    planner = TripPlanner()
    req = _request()
    plan = asyncio.run(planner.plan_trip(req))
    revised = asyncio.run(planner.revise_trip(req, plan, "把第一天改成去颐和园"))
    assert isinstance(revised, TripPlan)
