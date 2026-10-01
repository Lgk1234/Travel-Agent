"""Planning 节点：planner / weather / attraction / route / hotel / meal / budget / summary。

- planner_node、summary_node 会调用 LLM（各一次）。
- weather_node / attraction_node / route_node / hotel_node / meal_node / budget_node 只调用 tools，不调用 LLM。

注意：这里不做 `from ... import weather_node` 之类的重导出 —— 函数名与模块名同名会
覆盖子模块属性（`src.planning.nodes.weather_node` 变成函数），影响测试与按模块 mock。
使用处请直接 `from src.planning.nodes.weather_node import weather_node`。
"""
