"""Planning 图编排与对外入口（行程规划页）。

链路（严格按节点顺序）：
    planner_node(LLM) → weather → attraction → route → hotel → meal → budget → summary_node(LLM)

- planner_node / summary_node 各调用一次 LLM（共 2 次），负责「需求理解+任务计划」与「汇总成 TripPlan」。
- weather / attraction / route / hotel / meal / budget 只调用 tools，不调用 LLM。

节点通过 functools.partial 绑定请求上下文 PlanningContext（不用全局变量传请求态）。
"""
import logging
from functools import partial
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.llm import get_llm
from src.prompts.prompts import PLANNER_AGENT_PROMPT
from src.observability import LocalTraceCollector, run_config
from src.planning.nodes.attraction_node import attraction_node
from src.planning.nodes.budget_node import budget_node
from src.planning.nodes.hotel_node import hotel_node
from src.planning.nodes.meal_node import meal_node
from src.planning.nodes.planner_node import planner_node
from src.planning.nodes.route_node import route_node
from src.planning.nodes.summary_node import summary_node
from src.planning.nodes.weather_node import weather_node
from src.planning.state import AgentState, PlanningContext, ProgressCallback, TaskPlan
from src.schemas.trip import TripPlan, TripRequest
from src.tools.amap.client import get_amap_tools

logger = logging.getLogger("planning.graph")


def _request_text(request: TripRequest) -> str:
    """把结构化表单整理给 LLM 的文本。"""
    prefs = "、".join(request.preferences) if request.preferences else "无特殊偏好"
    budget = f"{request.budget} 元" if request.budget else "不限"
    return (
        f"目的地：{request.city}\n"
        f"出发日期：{request.start_date}，返程日期：{request.end_date}，共 {request.travel_days} 天\n"
        f"出行人数：{request.people_count} 人\n"
        f"预算：{budget}\n"
        f"交通方式：{request.transportation}\n"
        f"住宿类型：{request.accommodation}\n"
        f"偏好：{prefs}\n"
        f"补充需求：{request.free_text_input or '无'}"
    )


def _default_keywords(request: TripRequest) -> list[str]:
    """修订场景的兜底检索关键词（不额外调用 LLM）。"""
    kws = [p.strip() for p in (request.preferences or []) if p and p.strip()]
    free = (request.free_text_input or "").strip()
    if free:
        kws.append(free)
    return kws or [f"{request.city}热门景点"]


class TripPlanner:
    """行程规划编排器：加载能力 → 按请求构建图 → 执行 → 返回 TripPlan。"""

    def __init__(self):
        self.llm = None
        self.tools = None

    async def initialize(self, model: Optional[str] = None):
        if self.llm is not None:
            return
        self.llm = get_llm(model, disable_thinking=True)
        self.tools = await get_amap_tools()
        logger.info("行程规划初始化完成（工具数：%d）", len(self.tools or []))

    async def ensure_initialized(self, model: Optional[str] = None):
        await self.initialize(model)

    def _build_context(self, request: TripRequest, req_text: str,
                       on_progress: Optional[ProgressCallback]) -> PlanningContext:
        llm = self.llm
        return PlanningContext(
            request=request,
            req_text=req_text,
            llm=llm,
            planner_llm=llm.with_structured_output(TaskPlan),
            summary_llm=llm.with_structured_output(TripPlan),
            on_progress=on_progress,
        )

    def _build_graph(self, ctx: PlanningContext):
        builder = StateGraph(AgentState)
        builder.add_node("planner", partial(planner_node, ctx=ctx))
        builder.add_node("weather", partial(weather_node, ctx=ctx))
        builder.add_node("attraction", partial(attraction_node, ctx=ctx))
        builder.add_node("route", partial(route_node, ctx=ctx))
        builder.add_node("hotel", partial(hotel_node, ctx=ctx))
        # 餐饮检索（确定性）：为三餐提供真实店名，避免「XX周边餐厅」这类兜底写法
        builder.add_node("meal", partial(meal_node, ctx=ctx))
        # 预算参考先算（确定性），summary 读取后再填单价与 budget
        builder.add_node("budget", partial(budget_node, ctx=ctx))
        builder.add_node("summary", partial(summary_node, ctx=ctx))

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "weather")
        builder.add_edge("weather", "attraction")
        builder.add_edge("attraction", "route")
        builder.add_edge("route", "hotel")
        builder.add_edge("hotel", "meal")
        builder.add_edge("meal", "budget")
        builder.add_edge("budget", "summary")
        builder.add_edge("summary", END)
        return builder.compile()

    async def plan_trip(self, request: TripRequest,
                        on_progress: Optional[ProgressCallback] = None) -> TripPlan:
        await self.initialize()
        req_text = _request_text(request)
        ctx = self._build_context(request, req_text, on_progress)
        if on_progress:
            await on_progress("init", "正在分析你的旅行需求…")

        graph = self._build_graph(ctx)
        # 本地 trace 落盘：收集本次规划所有节点/工具调用及耗时
        local = LocalTraceCollector("planner")
        local.activate()
        try:
            result = await graph.ainvoke(
                {"req_text": req_text, "collected": {}, "plan": {}},
                config=run_config("orchestration"),
            )
        finally:
            local.deactivate()
            local.dump()

        plan = result["collected"].get("trip_plan")
        if plan is None:
            raise RuntimeError("行程规划失败：summary 未产出结果")
        if on_progress:
            await on_progress("done", "行程生成完成")
        return plan

    async def revise_trip(self, request: TripRequest, plan: TripPlan, instruction: str,
                          on_progress: Optional[ProgressCallback] = None) -> TripPlan:
        """按用户修改意见修订行程：默认只做 1 次 LLM 重汇总；
        仅当意见明确需要新数据时，才针对性重跑对应取数 node（仍不经 LLM）。
        """
        await self.initialize()
        req_text = _request_text(request)
        ctx = self._build_context(request, req_text, on_progress)

        local = LocalTraceCollector("revise")
        local.activate()
        if on_progress:
            await ctx.emit("init", "正在根据你的意见调整行程…")

        need_attraction = any(
            k in instruction
            for k in ("景点", "玩", "去", "博物馆", "美术馆", "公园", "美食", "加", "增", "删", "去掉", "换")
        )
        need_hotel = any(k in instruction for k in ("酒店", "住", "民宿", "新酒店"))

        fresh_blocks: list[str] = []
        if need_attraction:
            res = await attraction_node(
                {"req_text": req_text, "collected": {},
                 "plan": {"attraction_keywords": _default_keywords(request)}},
                ctx=ctx,
            )
            fresh_blocks.append("【景点最新工具结果】\n" + (res.get("collected", {}).get("attraction") or ""))
        if need_hotel:
            res = await hotel_node(
                {"req_text": req_text, "collected": {},
                 "plan": {"hotel_keyword": request.accommodation or "酒店"}},
                ctx=ctx,
            )
            fresh_blocks.append("【住宿最新工具结果】\n" + (res.get("collected", {}).get("hotel") or ""))
        if fresh_blocks and on_progress:
            await ctx.emit("revise_data", "正在更新相关信息…")

        synthesis = (
            f"用户原始需求：\n{req_text}\n\n"
            f"当前行程（JSON）：\n{plan.model_dump_json(indent=2)}\n\n"
            f"用户修改意见：{instruction}\n\n"
            + "\n\n".join(fresh_blocks)
            + "\n\n请基于修改意见修订行程：保持未被要求修改的部分不变，仅按意见调整，"
              "并输出完整、符合系统提示词结构的 JSON（酒店价格不得虚构）。"
        )
        try:
            revised: TripPlan = await ctx.summary_llm.ainvoke(
                [
                    SystemMessage(content=PLANNER_AGENT_PROMPT),
                    HumanMessage(content=synthesis),
                ],
                config=run_config("revise"),
            )
        finally:
            local.deactivate()
            local.dump()
        if on_progress:
            await ctx.emit("done", "行程修订完成")
        return revised
