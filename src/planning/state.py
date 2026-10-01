"""Planning 状态与任务计划模型。"""
from typing import Annotated, Any, Awaitable, Callable, List, Optional, TypedDict

from pydantic import BaseModel, Field

from src.schemas.trip import TripRequest

# 进度回调：on_progress(stage, detail)
ProgressCallback = Callable[[str, str], Awaitable[None]]


def merge_collected(left, right):
    """collected 字段的 reducer：把节点返回的新增产出合并进已有字典。"""
    if not left:
        left = {}
    if right:
        left.update(right)
    return left


class TaskPlan(BaseModel):
    """planner_node（LLM）产出的任务计划：指导各取数 node 查什么。

    LLM 只在这里做一次「需求理解 + 任务分解」，取数 node 之后不再调用 LLM。
    """
    attraction_keywords: List[str] = Field(
        default_factory=list, description="景点检索关键词（数量请控制在 6 个以内）"
    )
    hotel_keyword: str = Field(default="", description="住宿检索关键词")
    meal_keywords: List[str] = Field(
        default_factory=list,
        description="餐饮检索关键词（数量请控制在 3 个以内，如「北京烤鸭」「北京小吃街」）",
    )
    route_requirements: str = Field(default="", description="路线约束，如偏好交通方式/重点接驳段")
    reason: str = Field(default="", description="对用户需求的理解与规划理由")


class AgentState(TypedDict):
    """Planning 图的共享状态。"""
    req_text: str
    collected: Annotated[dict, merge_collected]  # weather/attraction/route/hotel 文本
    plan: dict                                    # TaskPlan 序列化结果


class PlanningContext:
    """各 node 共享的运行上下文（请求、模型、进度回调）。

    通过 functools.partial 绑定给 node，避免用全局变量传递请求态。
    """

    def __init__(self, request: TripRequest, req_text: str, llm: Any,
                 planner_llm: Any, summary_llm: Any,
                 on_progress: Optional[ProgressCallback] = None):
        self.request = request
        self.req_text = req_text
        self.llm = llm
        self.planner_llm = planner_llm
        self.summary_llm = summary_llm
        self.on_progress = on_progress

    async def emit(self, stage: str, detail: str):
        if self.on_progress:
            await self.on_progress(stage, detail)
