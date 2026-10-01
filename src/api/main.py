"""FastAPI 入口：路由、CORS、智能体单例与前端托管。

- 启动时预热智能体与高德 MCP 连接（单例）。
- /api/chat：对话问答，SSE 流式返回 token。
- /api/trip/plan：行程规划，SSE 返回进度事件 + 最终结构化结果。
- /：托管 frontend/index.html 双页面前端。
"""
import sys
from pathlib import Path

# 支持直接运行本文件（python src/api/main.py）：
# 直接执行时 sys.path 只含 src/api，这里把项目根目录补进去，保证 `from src...` 可导入。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from src.agents.chat_agent import ChatAgent
from src.observability import log_status
from src.planning.graph import TripPlanner
from src.schemas.chat import ChatRequest
from src.schemas.trip import TripRequest, TripReviseRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("api")

FRONTEND_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
)

chat_agent = ChatAgent()
planner = TripPlanner()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("应用启动，预热智能体与高德 MCP 连接...")
    log_status()
    try:
        await chat_agent._ensure_tools()
        await planner.ensure_initialized()
        logger.info("智能体初始化完成。")
    except Exception as e:  # noqa: BLE001
        logger.warning("启动时初始化出错（将在请求时降级处理）：%s", e)
    yield
    logger.info("应用关闭。")


app = FastAPI(title="智能出行助手", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """对话问答：SSE 流式返回助手文本片段。"""

    async def event_gen():
        yield {"event": "message", "data": json.dumps({"type": "start"}, ensure_ascii=False)}
        try:
            async for token in chat_agent.astream(request):
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {"type": "token", "content": token}, ensure_ascii=False
                    ),
                }
        except Exception as e:  # noqa: BLE001
            logger.exception("对话生成失败")
            yield {
                "event": "message",
                "data": json.dumps(
                    {"type": "error", "message": str(e)}, ensure_ascii=False
                ),
            }
        yield {"event": "message", "data": json.dumps({"type": "end"}, ensure_ascii=False)}

    return EventSourceResponse(event_gen())


@app.post("/api/trip/plan")
async def plan_trip(request: TripRequest):
    """行程规划：SSE 返回进度事件与最终结果。"""

    queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(stage: str, detail: str):
        await queue.put({"type": "progress", "stage": stage, "detail": detail})

    async def run():
        try:
            plan = await planner.plan_trip(request, on_progress=on_progress)
            result = plan.model_dump()
            await queue.put({"type": "result", "data": result})
        except Exception as e:  # noqa: BLE001
            logger.exception("行程规划失败")
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(None)

    async def event_gen():
        asyncio.create_task(run())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield {
                "event": "message",
                "data": json.dumps(item, ensure_ascii=False),
            }

    return EventSourceResponse(event_gen())


@app.post("/api/trip/revise")
async def revise_trip(request: TripReviseRequest):
    """行程修订（Human-in-the-loop）：根据修改意见重新规划。"""
    queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(stage: str, detail: str):
        await queue.put({"type": "progress", "stage": stage, "detail": detail})

    async def run():
        try:
            plan = await planner.revise_trip(
                request.request, request.plan, request.instruction, on_progress=on_progress
            )
            result = plan.model_dump()
            await queue.put({"type": "result", "data": result})
        except Exception as e:  # noqa: BLE001
            logger.exception("行程修订失败")
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(None)

    async def event_gen():
        asyncio.create_task(run())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield {"event": "message", "data": json.dumps(item, ensure_ascii=False)}

    return EventSourceResponse(event_gen())


@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
