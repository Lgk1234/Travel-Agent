"""budget_node：预算参考计算（**确定性，不调用 LLM**，位于 summary_node 之前）。

为什么要放在 summary 之前：预算本质是「可计算的结构化数据」。此前让 LLM 顺手填写，
结果全是 0（"预算当摆设"）。改为先由本节点基于：
  - 用户填写的总预算（未填则按人均每日标准估算）
  - 人数 / 天数 / 交通方式
  - 已检索到的景点、酒店数量
算出「总额 + 四项金额 + 各项参考单价」，写入 collected["budget"]，
供 summary_node 读取后据此填写各单价与 budget 字段。
"""
import logging

from src.planning.state import AgentState, PlanningContext
from src.schemas.trip import TripRequest

logger = logging.getLogger("planning.nodes.budget")

# 用户未填预算时的默认标准（元/人/天）
DEFAULT_PER_PERSON_PER_DAY = 500

# 参考拆分比例：住宿 / 餐饮 / 交通 / 门票（按交通方式微调，四项和为 1）
_RATIOS = {
    "自驾": {"hotel": 0.32, "meal": 0.28, "transport": 0.28, "ticket": 0.12},
    "公共交通": {"hotel": 0.35, "meal": 0.32, "transport": 0.15, "ticket": 0.18},
    "地铁": {"hotel": 0.35, "meal": 0.32, "transport": 0.15, "ticket": 0.18},
    "步行": {"hotel": 0.38, "meal": 0.35, "transport": 0.08, "ticket": 0.19},
    "_default": {"hotel": 0.35, "meal": 0.30, "transport": 0.20, "ticket": 0.15},
}


def _count_items(text: str) -> int:
    """粗略统计 POI 文本里的条目数（行首 `- `）。"""
    if not text:
        return 0
    return sum(1 for line in text.splitlines() if line.strip().startswith("- "))


def compute_budget_reference(request: TripRequest, attraction_count: int = 0,
                             hotel_count: int = 0) -> dict:
    """确定性计算预算参考：总额、四项金额与各项参考单价。"""
    people = max(int(request.people_count or 1), 1)
    days = max(int(request.travel_days or 1), 1)
    nights = max(days - 1, 1)

    user_budget = request.budget if (request.budget and request.budget > 0) else None
    total = int(user_budget) if user_budget else DEFAULT_PER_PERSON_PER_DAY * people * days

    ratios = _RATIOS.get((request.transportation or "").strip(), _RATIOS["_default"])
    hotel_total = int(round(total * ratios["hotel"]))
    meal_total = int(round(total * ratios["meal"]))
    transport_total = int(round(total * ratios["transport"]))
    ticket_total = int(round(total * ratios["ticket"]))

    meals_count = days * 3 * people  # 每天早中晚 × 人数
    attractions_est = max(attraction_count, days * 2)

    return {
        "total": total,
        "user_budget": user_budget,
        "people": people,
        "days": days,
        "nights": nights,
        "total_hotels": hotel_total,
        "total_meals": meal_total,
        "total_transportation": transport_total,
        "total_attractions": ticket_total,
        "hotel_per_night": int(round(hotel_total / nights)),
        "meal_per_person": int(round(meal_total / meals_count)),
        "transport_per_person_day": int(round(transport_total / (days * people))),
        "ticket_per_person": int(round(ticket_total / (attractions_est * people))),
        "attractions_est": attractions_est,
        "hotels_found": hotel_count,
    }


def format_budget_reference(ref: dict) -> str:
    """把预算参考格式化为给 LLM 阅读的文本。"""
    src = (f"用户填写 {ref['user_budget']} 元" if ref["user_budget"]
           else f"未填写，按人均 {DEFAULT_PER_PERSON_PER_DAY} 元/天估算")
    return "\n".join([
        f"【预算参考】（总额 {ref['total']} 元，{ref['people']} 人 {ref['days']} 天；来源：{src}）",
        f"- 住宿：{ref['total_hotels']} 元（约 {ref['hotel_per_night']} 元/晚，共 {ref['nights']} 晚）",
        f"- 餐饮：{ref['total_meals']} 元（约 {ref['meal_per_person']} 元/人/餐）",
        f"- 交通：{ref['total_transportation']} 元（约 {ref['transport_per_person_day']} 元/人/天）",
        f"- 门票：{ref['total_attractions']} 元（约 {ref['ticket_per_person']} 元/人/景点，"
        f"按 {ref['attractions_est']} 个景点估）",
        "请据此为每天景点/餐饮/酒店填写参考单价，并填写 budget 各项与 total（**禁止全 0**）；"
        "所有价格均为参考价，请在 description 注明「参考价，以实际预订为准」。",
    ])


async def budget_node(state: AgentState, *, ctx: PlanningContext) -> dict:
    await ctx.emit("budget", "正在估算预算分配…")
    collected = state.get("collected") or {}
    attraction_count = _count_items(collected.get("attraction", ""))
    hotel_count = _count_items(collected.get("hotel", ""))
    ref = compute_budget_reference(ctx.request, attraction_count, hotel_count)
    logger.info("预算参考计算完成：total=%s（景点条目 %d，酒店条目 %d）",
                ref["total"], attraction_count, hotel_count)
    return {"collected": {"budget": format_budget_reference(ref)}}
