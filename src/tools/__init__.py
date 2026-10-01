"""Tool 能力层：统一封装外部能力接入（高德 MCP、和风天气）。

- 高德：单一 MCP 连接，按业务能力分组为 tools/amap/{client,poi,route,hotel}。
- 和风：独立 REST API，tools/weather_tool.py（GeoAPI 定位 + Weather API 查询）。
本层只提供能力，不含业务编排；由 Agent 层与 Planning 的 node 直接调用。
"""
