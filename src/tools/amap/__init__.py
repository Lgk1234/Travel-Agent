"""高德地图能力（单一 MCP 连接，按业务能力分组）。

- `client.py`：唯一 MCP client 初始化 + 工具列表 + 缓存/限流/重试/结果压缩（**唯一建连处**）。
- `poi.py` / `route.py` / `hotel.py`：基于 client 的业务能力封装，**不重复创建连接**。
"""
