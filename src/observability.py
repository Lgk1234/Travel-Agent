"""可观测性（Service 基础层）：本地 trace 落盘 + LangSmith 云端追踪（可选）。

合并自原可观测模块，职责不变：

1. **本地 trace 落盘**（诊断主力）
   - 为什么需要：LangSmith 云端 trace 需手动导出，且常是单行超大 JSON（曾 216KB 无计时），
     难以定位「慢在哪」。本模块在请求根调用挂一个 LangChain 回调，请求结束后把整条调用树
     写成轻量 JSON 落到 ./traces/：自带每节点 latency_ms、token 用量、工具名，并截断超大返回。
   - 通过 contextvars 做请求级隔离：并发请求各自 handler 互不污染。
   - `run_config()` 在存在 active handler 时自动注入 callbacks，因此 graph / 子节点 /
     工具调用（含嵌套）全部被同一 handler 收集。
   - 全程 best-effort：任何异常都不影响主流程。

2. **LangSmith 云端追踪**（可选）
   - .env 设置 LANGSMITH_API_KEY + LANGSMITH_TRACING 后自动上报，无需额外代码。
   - 无 key 时全部降级为本地日志，不影响功能。
"""
import contextvars
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.base import AsyncCallbackHandler

from src.config import LANGSMITH_API_KEY, LANGSMITH_TRACING

logger = logging.getLogger("observability")

PROJECT = "smart-travel-agent"

TRACE_DIR = os.getenv("LOCAL_TRACE_DIR", "traces")
# 单个 inputs/outputs 字段最大字符数，超过截断，防止路线/geo 等大返回撑爆文件
_MAX_FIELD_CHARS = int(os.getenv("LOCAL_TRACE_MAX_FIELD", "800"))

# 请求级当前生效的 collector（contextvars，天然按 async 任务隔离）
_active_handler: "contextvars.ContextVar[Optional['LocalTraceCollector']]" = (
    contextvars.ContextVar("local_trace_handler", default=None)
)


def active_local_trace() -> Optional["LocalTraceCollector"]:
    """返回当前请求上下文里生效的 collector（供 run_config 注入 callbacks）。"""
    return _active_handler.get()


def _truncate(obj: Any, max_chars: int = _MAX_FIELD_CHARS) -> Optional[str]:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        try:
            s = str(obj)
        except Exception:  # noqa: BLE001
            s = "<unserializable>"
    if len(s) > max_chars:
        s = s[:max_chars] + "...(truncated)"
    return s


def _extract_token_usage(response: Any) -> Optional[Dict[str, Any]]:
    """从 LLM / ChatModel 返回里抽 token 用量。

    兼容多种形态：
    - 单条 AIMessage / ChatGeneration
    - LLMResult（含 llm_output.token_usage）
    - chat_model_end 实际收到的嵌套列表 [[ChatGeneration, ...]]
    优先使用 usage_metadata，其次 response_metadata.usage（DashScope 常见）。
    """
    result: Dict[str, Any] = {}

    def _merge_openai_usage(u: Any):
        if not isinstance(u, dict):
            return
        mapping = {
            "prompt_tokens": "input_tokens",
            "completion_tokens": "output_tokens",
            "total_tokens": "total_tokens",
        }
        for ok, nk in mapping.items():
            if u.get(ok) is not None and nk not in result:
                result[nk] = u[ok]
        rd = u.get("completion_tokens_details") or u.get("output_token_details")
        if isinstance(rd, dict) and rd.get("reasoning_tokens") is not None and "reasoning" not in result:
            result["reasoning"] = rd["reasoning_tokens"]

    def _scan(obj):
        # 递归展平列表/元组（如 chat_model_end 收到的 [[ChatGeneration, ...]]）
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _scan(item)
            return
        if obj is None:
            return
        # ChatResult / LLMResult：逐 generation 展开
        gens = getattr(obj, "generations", None)
        if isinstance(gens, list):
            for g in gens:
                _scan(g)
            return
        # ChatGeneration 包了一层 message，真正 usage 在 message 上
        msg = getattr(obj, "message", None)
        if msg is not None:
            _scan(msg)
            return
        if not hasattr(obj, "usage_metadata") and not hasattr(obj, "response_metadata") \
                and not hasattr(obj, "llm_output"):
            return
        # 1) usage_metadata
        um = getattr(obj, "usage_metadata", None)
        if isinstance(um, dict):
            for k in ("input_tokens", "output_tokens", "total_tokens", "cache_read", "cache_creation"):
                if um.get(k) is not None:
                    result[k] = um[k]
            det = um.get("output_token_details") or um.get("input_token_details")
            if isinstance(det, dict) and det.get("reasoning") is not None and "reasoning" not in result:
                result["reasoning"] = det["reasoning"]
        elif hasattr(um, "model_dump"):
            try:
                d = um.model_dump()
            except Exception:  # noqa: BLE001
                d = None
            if isinstance(d, dict):
                for k in ("input_tokens", "output_tokens", "total_tokens"):
                    if d.get(k) is not None:
                        result[k] = d[k]
        # 2) response_metadata.usage（DashScope 常见）
        rm = getattr(obj, "response_metadata", None)
        if isinstance(rm, dict):
            _merge_openai_usage(rm.get("usage") or rm.get("token_usage") or rm.get("tokenUsage"))
        # 3) llm_output（LLM 类型）
        lo = getattr(obj, "llm_output", None)
        if isinstance(lo, dict) and isinstance(lo.get("token_usage"), dict):
            _merge_openai_usage(lo["token_usage"])

    _scan(response)
    return result or None


class LocalTraceCollector(AsyncCallbackHandler):
    """收集一次请求内所有 LangChain/LangGraph run，结束后落盘为轻量 trace。"""

    def __init__(self, agent_name: str, trace_dir: str = TRACE_DIR):
        self.agent_name = agent_name
        self.trace_dir = trace_dir
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._root_ids: List[str] = []
        self._token = None
        self._dumped = False

    # ---- 请求级上下文管理（contextvars 隔离）----
    def activate(self) -> "LocalTraceCollector":
        self._token = _active_handler.set(self)
        return self

    def deactivate(self) -> None:
        if self._token is not None:
            _active_handler.reset(self._token)
            self._token = None

    # ---- 内部记录 ----
    def _rid(self, run_id) -> str:
        return str(run_id)

    def _on_start(self, run_id, parent_run_id, name, rtype, inputs=None):
        rid = self._rid(run_id)
        self._runs[rid] = {
            "id": rid,
            "name": name or rtype,
            "run_type": rtype,
            "parent_run_id": self._rid(parent_run_id) if parent_run_id else None,
            "start": time.time(),
            "end": None,
            "latency_ms": None,
            "inputs": _truncate(inputs),
            "outputs": None,
            "token_usage": None,
            "error": None,
            "children": [],
        }
        if not parent_run_id:
            self._root_ids.append(rid)

    def _on_end(self, run_id, outputs=None, token_usage=None):
        rec = self._runs.get(self._rid(run_id))
        if rec is None:
            return
        rec["end"] = time.time()
        rec["latency_ms"] = round((rec["end"] - rec["start"]) * 1000, 1)
        if outputs is not None:
            rec["outputs"] = _truncate(outputs)
        if token_usage is not None:
            rec["token_usage"] = token_usage

    def _on_error(self, run_id, error):
        rec = self._runs.get(self._rid(run_id))
        if rec is not None:
            rec["error"] = _truncate(str(error))

    # ---- LangChain 回调 ----
    async def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, name=None, **kwargs):
        self._on_start(run_id, parent_run_id, name, "chain", inputs)

    async def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._on_end(run_id, outputs=outputs)

    async def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._on_error(run_id, error)

    async def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, name=None, **kwargs):
        self._on_start(run_id, parent_run_id, name, "llm", prompts)

    async def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self._on_end(run_id, outputs=getattr(response, "generations", None) or response,
                     token_usage=_extract_token_usage(response))

    async def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._on_error(run_id, error)

    async def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, name=None, **kwargs):
        self._on_start(run_id, parent_run_id, name, "chat_model", messages)

    async def on_chat_model_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self._on_end(run_id, token_usage=_extract_token_usage(response))

    async def on_chat_model_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._on_error(run_id, error)

    async def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, name=None, **kwargs):
        # 部分调用链不会传 name，从 serialized 兜底取，保证 trace 里能看到真实工具名
        tool_name = name or (serialized or {}).get("name")
        self._on_start(run_id, parent_run_id, tool_name, "tool", input_str)

    async def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self._on_end(run_id, outputs=output)

    async def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._on_error(run_id, error)

    async def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, name=None, **kwargs):
        self._on_start(run_id, parent_run_id, name, "retriever", query)

    async def on_retriever_end(self, documents, *, run_id, parent_run_id=None, **kwargs):
        self._on_end(run_id, outputs=documents)

    async def on_retriever_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._on_error(run_id, error)

    # ---- 落盘 ----
    def _build_node(self, rid: str, seen: set):
        rec = self._runs.get(rid)
        if rec is None or rid in seen:
            return None
        seen.add(rid)
        node = {k: v for k, v in rec.items() if k not in ("parent_run_id", "start", "end")}
        children = [c for c in self._runs.values() if c.get("parent_run_id") == rid and c["id"] != rid]
        node["children"] = [self._build_node(c["id"], seen) for c in children]
        return node

    def dump(self) -> None:
        if self._dumped:
            return
        self._dumped = True
        try:
            os.makedirs(self.trace_dir, exist_ok=True)
            roots = []
            for rid in self._root_ids:
                n = self._build_node(rid, set())
                if n:
                    roots.append(n)

            total_tokens = 0
            n_tools = 0
            for r in self._runs.values():
                if r["run_type"] == "tool":
                    n_tools += 1
                tu = r.get("token_usage")
                if isinstance(tu, dict):
                    try:
                        total_tokens += int(tu.get("total_tokens") or tu.get("total") or 0)
                    except (TypeError, ValueError):
                        pass
            root_latency = max((r["latency_ms"] or 0) for r in roots) if roots else 0

            payload = {
                "agent": self.agent_name,
                "dumped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "root_latency_ms": root_latency,
                "run_count": len(self._runs),
                "tool_call_count": n_tools,
                "total_tokens": total_tokens,
                "runs": roots,
            }

            latest_path = os.path.join(self.trace_dir, "latest.json")
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            ts = time.strftime("%Y%m%d_%H%M%S")
            rid8 = (self._root_ids[0][:8] if self._root_ids else "x")
            hist_path = os.path.join(self.trace_dir, f"{self.agent_name}_{ts}_{rid8}.json")
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            logger.info("本地 trace 已落盘：%s（根耗时 %.0fms，%d runs，%d 工具调用）",
                        latest_path, root_latency, len(self._runs), n_tools)
        except Exception as e:  # noqa: BLE001
            logger.warning("本地 trace 落盘失败（不影响主流程）：%s", e)


# ---- LangSmith 云端追踪（可选）----
def is_tracing_enabled() -> bool:
    return bool(LANGSMITH_API_KEY and LANGSMITH_TRACING)


def run_config(agent_name: str) -> Dict[str, Any]:
    """返回注入到 agent 调用 config 中的元数据/标签。

    - 云端追踪开启时注入 metadata/tags；
    - 若当前请求存在本地 trace collector，一并注入 callbacks，使本次请求内所有嵌套 run
      （含子智能体/工具）都被收集落盘。
    无 key 且无可收集 handler 时返回空 dict，不影响功能。
    """
    cfg: Dict[str, Any] = {}
    if is_tracing_enabled():
        cfg["metadata"] = {"agent": agent_name, "app": PROJECT}
        cfg["tags"] = [PROJECT, agent_name]
    # 本地 trace：请求级 collector（contextvars 隔离），不依赖云端是否开启
    handler = active_local_trace()
    if handler is not None:
        cfg["callbacks"] = [handler]
    return cfg


def log_status() -> None:
    if is_tracing_enabled():
        logger.info("LangSmith 链路追踪已启用（project=%s）", PROJECT)
    else:
        logger.info(
            "LangSmith 未启用，使用本地日志（设置 LANGSMITH_API_KEY+LANGSMITH_TRACING 可开启）"
        )
