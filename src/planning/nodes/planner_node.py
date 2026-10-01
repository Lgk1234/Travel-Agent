"""planner_node：LLM 调用点 1 —— 需求理解 + 生成任务计划。

只在这一步用 LLM 做「理解与分解」，之后取数 node 全部确定性执行；
LLM 失败时回退为基于 TripRequest 的确定性计划，保证流程不中断。
"""
import logging

from langchain_core.messages import HumanMessage

from src.observability import run_config
from src.planning.state import AgentState, PlanningContext, TaskPlan
from src.prompts.prompts import PLANNER_PLAN_PROMPT
from src.tools.amap.poi import MAX_SEARCHES

logger = logging.getLogger("planning.nodes.planner")

MAX_KEYWORDS = MAX_SEARCHES  # 景点检索关键词上限，防止工具调用爆炸


def _fallback_plan(ctx: PlanningContext) -> TaskPlan:
    """planner LLM 不可用时，基于请求字段确定性推导检索计划。"""
    req = ctx.request
    kws: list[str] = []
    for p in (req.preferences or []):
        if p and p.strip():
            kws.append(p.strip())
    free = (req.free_text_input or "").strip()
    if free:
        kws.append(free)
    if not kws:
        kws = [f"{req.city}热门景点"]
    return TaskPlan(
        attraction_keywords=kws[:MAX_KEYWORDS],
        hotel_keyword=(req.accommodation or "酒店").strip(),
        route_requirements=(req.transportation or "").strip(),
        reason="（planner 回退：基于请求字段推导）",
    )


async def planner_node(state: AgentState, *, ctx: PlanningContext) -> dict:
    await ctx.emit("planning", "正在理解你的需求并制定检索计划…")
    try:
        plan: TaskPlan = await ctx.planner_llm.ainvoke(
            [HumanMessage(content=PLANNER_PLAN_PROMPT + "\n\n用户需求：\n" + ctx.req_text)],
            config=run_config("planner"),
        )
        if plan is None:
            raise ValueError("planner 未产出任务计划")
    except Exception as e:  # noqa: BLE001
        logger.warning("planner_node 失败，回退确定性计划：%s", e)
        plan = _fallback_plan(ctx)

    # 兜底与收敛：关键词去空、去重、限上限；住宿关键词兜底
    kws: list[str] = []
    for k in (plan.attraction_keywords or []):
        k = (k or "").strip()
        if k and k not in kws:
            kws.append(k)
    if not kws:
        kws = _fallback_plan(ctx).attraction_keywords
    plan.attraction_keywords = kws[:MAX_KEYWORDS]
    if not (plan.hotel_keyword or "").strip():
        plan.hotel_keyword = (ctx.request.accommodation or "酒店").strip()

    await ctx.emit("planning", f"检索计划：{'、'.join(plan.attraction_keywords)}")
    return {"plan": plan.model_dump()}
