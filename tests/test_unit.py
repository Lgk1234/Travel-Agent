"""单元测试：接入层解析、状态 reducer、路线 node 辅助函数、LLM 工厂、提示词、Schema。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm import get_llm
from src.planning.nodes.route_node import _first_location, _names_from_poi
from src.planning.state import merge_collected
from src.prompts.prompts import (
    CHAT_AGENT_PROMPT,
    PLANNER_AGENT_PROMPT,
    PLANNER_PLAN_PROMPT,
)
from src.schemas.chat import ChatRequest
from src.schemas.trip import TripRequest
from src.tools.amap.client import _format_amap_weather, _payload
from src.tools.weather_tool import _needed_daily_days, _select_casts


def test_merge_collected_reducer():
    assert merge_collected(None, {"weather": "w"}) == {"weather": "w"}
    assert merge_collected({"a": "1"}, {"b": "2"}) == {"a": "1", "b": "2"}


def test_extract_json_payload_handles_content_block():
    """MCP content-block 形态（list 内含 {'text': '{json}'}）应被正确解析。"""
    raw = [{"type": "text", "text": '{"pois":[{"name":"故宫"}]}'}]
    assert _payload(raw) == {"pois": [{"name": "故宫"}]}


def test_format_amap_weather_parses_tuple_string():
    inner = (
        '{"status":"1","forecasts":[{"city":"北京市","casts":['
        '{"date":"2026-09-30","week":"3","dayweather":"晴","nightweather":"多云",'
        '"daytemp":"25","nighttemp":"14","daywind":"南","daypower":"1-3"}]}]}'
    )
    raw = '[("' + inner + '", None)]'
    out = _format_amap_weather("北京", raw)
    assert "北京市" in out
    assert "晴" in out


def test_select_casts_by_trip_dates():
    """行程 10-02~10-03 必须拿到这两天，而不是从今天起的前两天。"""
    casts = [{"fxDate": "2026-10-01"}, {"fxDate": "2026-10-02"}, {"fxDate": "2026-10-03"}]
    selected, note = _select_casts(casts, 2, "2026-10-02", "2026-10-03")
    assert [c["fxDate"] for c in selected] == ["2026-10-02", "2026-10-03"]
    assert note == ""


def test_select_casts_out_of_range_has_note():
    casts = [{"fxDate": "2026-10-01"}, {"fxDate": "2026-10-02"}]
    selected, note = _select_casts(casts, 2, "2026-11-01", "2026-11-03")
    assert len(selected) == 2
    assert "未覆盖行程日期" in note


def test_needed_daily_days():
    assert _needed_daily_days(2, "2027-01-01") == 7  # 远期行程升级 7d
    assert _needed_daily_days(2, "") == 3


def test_names_from_poi_takes_main_per_group():
    """每组只取主景点，避免拿到检票处/某某门等子点。"""
    from src.planning.nodes.route_node import _names_from_poi

    text = (
        "【北京 故宫博物院】\n- 故宫博物院｜｜景山前街4号\n- 故宫博物院检票处｜｜院内\n- 故宫博物院-午门｜｜院内\n"
        "【北京 天坛公园】\n- 天坛公园｜｜天坛东里甲1号\n- 天坛公园售票处｜｜园内"
    )
    assert _names_from_poi(text) == ["故宫博物院", "天坛公园"]


def test_payload_handles_tuple_with_json_string():
    """高德 MCP 会返回 ('{json}', None) 形态，必须能解析（否则地理编码压缩失效）。"""
    from src.tools.amap.client import _payload

    raw = ('{"results":[{"location":"116.1,39.9","level":"兴趣点"}]}', None)
    data = _payload(raw)
    assert isinstance(data, dict)
    assert data["results"][0]["location"] == "116.1,39.9"


def test_center_from_route():
    from src.planning.nodes.hotel_node import center_from_route

    text = "景点坐标：\n- 故宫｜116.397,39.908\n- 天坛｜116.410,39.882\n相邻接驳：…"
    center = center_from_route(text)
    lon, lat = center.split(",")
    assert abs(float(lon) - 116.4035) < 0.01
    assert abs(float(lat) - 39.895) < 0.01
    assert center_from_route("无坐标文本") == ""


def test_compute_budget_reference_with_user_budget():
    from src.planning.nodes.budget_node import compute_budget_reference, format_budget_reference
    from src.schemas.trip import TripRequest

    req = TripRequest(city="北京", start_date="2026-10-02", end_date="2026-10-03",
                      travel_days=2, people_count=2, budget=5000, transportation="自驾")
    ref = compute_budget_reference(req, attraction_count=4, hotel_count=3)
    assert ref["total"] == 5000
    # 四项之和等于总额
    assert (ref["total_hotels"] + ref["total_meals"]
            + ref["total_transportation"] + ref["total_attractions"]) == 5000
    assert ref["hotel_per_night"] > 0
    assert ref["meal_per_person"] > 0
    assert ref["transport_per_person_day"] > 0
    assert ref["ticket_per_person"] > 0
    text = format_budget_reference(ref)
    assert "预算参考" in text and "禁止全 0" in text


def test_compute_budget_reference_default_total():
    from src.planning.nodes.budget_node import compute_budget_reference
    from src.schemas.trip import TripRequest

    req = TripRequest(city="北京", start_date="2026-10-02", end_date="2026-10-03",
                      travel_days=2, people_count=2)
    ref = compute_budget_reference(req)
    # 未填预算：人均 500 元/天 × 2 人 × 2 天
    assert ref["total"] == 2000
    assert ref["user_budget"] is None


def test_route_node_helpers():
    text = "- 故宫博物院｜博物馆｜景山前街4号\n- 天安门广场｜景点｜长安街\n无关行"
    assert _names_from_poi(text) == ["故宫博物院", "天安门广场"]
    assert _first_location("某地址｜116.397,39.908｜兴趣点") == "116.397,39.908"
    assert _first_location("无坐标") == ""


def test_llm_factory_disable_thinking():
    llm = get_llm("qwen-test", disable_thinking=True)
    # enable_thinking=false 应注入到 extra_body 顶层参数
    assert getattr(llm, "extra_body", None) == {"enable_thinking": False}


def test_prompts_present():
    for p in (CHAT_AGENT_PROMPT, PLANNER_AGENT_PROMPT, PLANNER_PLAN_PROMPT):
        assert isinstance(p, str) and len(p) > 20


def test_schemas_validate():
    ChatRequest(message="hi", model="x")
    TripRequest(city="北京", start_date="2026-10-01", end_date="2026-10-03", travel_days=3)
