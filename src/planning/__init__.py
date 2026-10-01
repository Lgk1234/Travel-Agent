"""Planning 编排层（行程规划页面入口）。

流程：planner_node(LLM: 需求理解+任务计划) → weather → attraction → route → hotel → meal → budget
      → summary_node(LLM: 汇总成 TripPlan)

取数 node（weather/attraction/route/hotel/meal/budget）只调用 tools，不调用 LLM。
"""
