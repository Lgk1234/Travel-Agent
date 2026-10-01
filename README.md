# 智能出行助手 Agent

基于 LangChain / LangGraph 的出行助手，提供**两个独立页面、两个业务入口**：

- **出行问答**：聊天式问答，LLM + Tool Calling 回答天气 / 景点 / 路线等查询（天气走**和风天气 API**，地图走**高德 MCP**）。
- **行程规划**：填写目的地、日期、人数、预算、交通方式、偏好 → LangGraph 编排多个 node 抓取真实数据 → LLM 生成结构化行程。

> 两条业务流程分离；LLM 接入、工具、数据模型、可观测等基础能力共用。

## 核心功能

- **对话问答**：基于 ReAct + 工具调用，回答实时天气、地点搜索、路线规划等问题，SSE 流式输出。
- **行程规划**：8 节点流水线，抓取真实景点 / 天气 / 路线 / 酒店 / 餐饮数据，产出含每日安排、预算参考、接驳路线的结构化行程。
- **预算参考**：确定性节点按人数 / 天数 / 交通方式 / 已检索数据计算参考单价，避免价格全为 0。
- **酒店就近**：按景点群中心点周边检索住宿。
- **餐饮推荐**：基于真实 POI 的餐厅推荐，跨天去重与品类轮换。
- **天气对齐**：逐日预报按行程区间筛选，超出预报范围如实说明。
- **多轮记忆**：问答页基于 `thread_id` 的短期记忆。
- **降级与可观测**：工具异常返回降级文本（绝不编造数值），本地 trace 落盘便于排查。

## 技术栈

- 语言：Python 3.11+
- Web：FastAPI + SSE（`sse-starlette`）+ Uvicorn + Pydantic v2
- Agent / 编排：LangChain 1.x / LangGraph 1.x
- 模型网关：阿里云百炼 OpenAI 兼容接口（统一一个端点，模型名透传，关闭思考 `disable_thinking`）
- 地图能力：高德地图 MCP（`langchain-mcp-adapters`，单一连接）
- 天气：和风天气 REST API（GeoAPI + Weather API）
- 测试：pytest（离线 mock）

## 系统架构

四层结构，两个业务入口互不耦合：

```
Agent 层     src/agents/chat_agent.py   问答入口，LLM 直调 tools
Planning 层  src/planning/               LangGraph：planner → 取数 node → summary
Tool 能力层  src/tools/                  amap/{client,poi,route,hotel} + weather_tool
Service 层   src/{config,llm,observability}.py + schemas/ + prompts/
```

规划页节点链（严格顺序，**全程仅 2 次 LLM 调用**）：

```
planner_node(LLM1: 需求理解 + 任务计划)
  → weather_node    和风天气
  → attraction_node 高德 POI
  → route_node      高德路线 / 距离
  → hotel_node      高德住宿（按景点群就近）
  → meal_node       高德 POI 餐饮
  → budget_node     确定性预算参考（无 LLM）
  → summary_node(LLM2: 汇总 → 结构化 TripPlan)
```

取数 node（weather / attraction / route / hotel / meal / budget）只调 tools，不调 LLM；planner 与 summary 各调一次 LLM。

## 两个页面

- **问答页**（`frontend/index.html` 的对话区）：用户输入问题 → `/api/chat`（SSE 流式）→ Chat Agent（ReAct）判断并调用工具 → 实时返回文本片段。
- **规划页**（`frontend/index.html` 的规划区）：表单提交 → `/api/trip/plan`（SSE 进度 + 结构化结果）→ Planning Agent 跑完上述节点链 → 返回 TripPlan JSON。

## Agent / Planning / Tool / LangGraph / MCP 的关系

- **LangGraph** 是规划页的编排框架：`StateGraph` 把各 node 串成固定顺序的流水线，通过 `PlanningContext` 传递请求上下文（不用全局变量）。
- **Agent 层（Chat Agent）** 是问答页入口，使用 LangGraph 的 ReAct 预置 Agent，直接把工具交给 LLM 调度；它不经过 Planning 的节点链。
- **Planning 层** 的 `planner_node` 是一次性规划（产出任务计划：景点 / 住宿 / 餐饮 / 路线关键词），后续 node 确定性执行，不引入子 Agent。
- **Tool 能力层** 是统一外部能力：高德 MCP 与和风天气。node 和 Chat Agent 都通过 Tool 访问外部数据。
- **MCP** 是高德地图能力的统一接入方式（单一连接、缓存、限流、退避、结果压缩），对上层屏蔽底层 HTTP。

## 高德 MCP 的作用

提供统一的地图能力：POI 检索、路线 / 距离规划、地理编码、周边搜索。所有地图相关调用都经过 `tools/amap/client.py` 这一个 MCP 连接，避免重复建连；接入层做了 TTL 缓存、信号量限流、QPS 指数退避与结果压缩，既保护配额也避免大响应压垮上下文。

## 和风天气 API 的作用

提供天气数据，独立于 MCP（直接 REST）。先用 GeoAPI 把城市名解析为经纬度，再请求实况 / 逐日（3d / 7d）/ 生活指数。和风要求使用控制台生成的**独立 API Host** 与 `X-QW-Api-Key` 鉴权；Agent 不直接发 HTTP，只通过 `@tool` 封装的 `weather_tool` 使用，并带缓存与降级（失败回退高德或标注不可用）。

## 项目结构

```
src/
├── config.py / llm.py / observability.py   # Service 基础层
├── schemas/                                # chat.py、trip.py 数据模型
├── prompts/                                # 提示词
├── tools/
│   ├── amap/
│   │   ├── client.py   # 唯一高德 MCP 连接（缓存 / 限流 / 退避 / 压缩）
│   │   ├── poi.py      # POI 检索 / 地理编码 / 周边搜
│   │   ├── route.py    # 路线 / 距离
│   │   └── hotel.py    # 住宿检索（周边搜 / 民宿回退酒店）
│   └── weather_tool.py # 和风天气封装
├── agents/
│   └── chat_agent.py   # Chat Agent（问答页）
├── planning/
│   ├── state.py        # AgentState + TaskPlan + PlanningContext
│   ├── nodes/          # planner / weather / attraction / route / hotel / meal / budget / summary
│   └── graph.py        # StateGraph 编排
└── api/main.py         # 启动 / 路由 / SSE
tests/                  # pytest，离线 mock
frontend/index.html     # 双页面前端
```

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env      # 填入真实密钥（见下）
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# 浏览器打开 frontend/index.html（双页面）
```

## 环境变量配置（.env）

| 变量 | 说明 |
|---|---|
| `DEFAULT_MODEL` | 百炼模型名，可随时切换 |
| `LLM_API_KEY` | 百炼网关 Key |
| `LLM_BASE_URL` | 百炼网关 base_url |
| `AMAP_API_KEY` | 高德地图 Key（MCP 用） |
| `QWEATHER_API_HOST` | 和风「控制台生成的独立 API Host」（Geo 与 Weather 共用，仅路径不同） |
| `QWEATHER_API_KEY` | 和风 Key |
| `LANGSMITH_API_KEY` / `LANGSMITH_TRACING` | 可选，开启 LangSmith 云端追踪 |

> 和风注意：`QWEATHER_API_HOST` 必须填控制台生成的专属 Host；否则天气会回退高德或标注不可用。
> 真实密钥请写在本地 `.env`，不要提交到仓库（`.env` 已在 `.gitignore` 中）。

## 测试

```bash
python -m pytest tests/ -q
```

测试用 mock 屏蔽真实 MCP / 和风 / LLM，可完全离线运行；覆盖工具解析、reducer、node 辅助函数、LLM 工厂、提示词、Schema 与规划链路端到端。

## 设计取舍 / 未实现

为保持简单与可预测，当前**未实现**以下功能（避免夸大）：

- 长期记忆 / 用户偏好画像
- 评估-改进闭环（judge）
- RAG
- attraction / route 的 LLM 子智能体（采用 node 确定性调 tools）

所有价格均为**参考价**，以实际预订为准；工具不可用时如实标注，绝不编造数值。
