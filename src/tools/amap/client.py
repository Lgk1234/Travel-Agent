"""高德地图 MCP 客户端（单例）。

全局单例缓存，避免每次请求都重建 SSE 连接。高德 Key 缺失或连接失败时降级为「空工具列表」，
使 Agent 退化为纯模型生成并给出提示，而不是直接崩溃。
"""
import ast
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import AMAP_API_KEY

logger = logging.getLogger("amap_mcp")

# ---- 工具级缓存（层 A：减少重复高德调用 / 缓解 QPS 限流）----
_AMAP_CACHE: dict[str, tuple[float, str]] = {}
_AMAP_CACHE_TTL = float(os.getenv("AMAP_CACHE_TTL", "3600"))  # 默认 1 小时
_AMAP_CACHE_DISABLED = os.getenv("DISABLE_AMAP_CACHE", "").lower() in ("1", "true", "yes")

# ---- 高德 QPS 限流（治本：把并发突发平滑成 ≤ 配额 QPS）----
# 所有高德调用共用一把 key，route/attraction 并行 + 突发易触发
# CUQPS_HAS_EXCEEDED_THE_LIMIT。信号量限制并发；最小间隔保证 ≤ AMAP_MAX_QPS；
# 命中限流后指数退避重试（抖动防惊群）。与工具级缓存互补：缓存减少总次数，限流平滑突发。
_AMAP_MAX_CONCURRENCY = int(os.getenv("AMAP_MAX_CONCURRENCY", "4"))
_AMAP_MIN_INTERVAL = float(os.getenv("AMAP_MIN_INTERVAL", "0.22"))  # 间隔≥220ms ≈ ≤4.5 QPS
_AMAP_SEM = asyncio.Semaphore(_AMAP_MAX_CONCURRENCY)
_AMAP_RATE_LOCK = asyncio.Lock()
_AMAP_LAST_TS = 0.0


async def _throttled_call(coro, args, kwargs):
    """在信号量约束下执行调用，并对相邻调用施加最小间隔（平滑 QPS）。"""
    global _AMAP_LAST_TS
    async with _AMAP_SEM:
        async with _AMAP_RATE_LOCK:
            wait = _AMAP_MIN_INTERVAL - (time.monotonic() - _AMAP_LAST_TS)
            if wait > 0:
                await asyncio.sleep(wait)
            _AMAP_LAST_TS = time.monotonic()
        return await _invoke_coro(coro, args, kwargs)


async def _call_with_retry(coro, args, kwargs, max_retry: int = 3):
    """对 QPS 类瞬时错误做指数退避重试（抖动防惊群）。"""
    last = None
    for i in range(max_retry):
        try:
            return await _throttled_call(coro, args, kwargs)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if i < max_retry - 1 and ("QPS" in msg or "CUQPS" in msg or "限流" in msg):
                await asyncio.sleep(0.4 * (2 ** i) + random.random() * 0.2)
                last = e
                continue
            raise
    raise last


def _amap_cache_key(tool_name: str, args) -> str:
    try:
        arg_str = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        arg_str = str(args)
    return hashlib.md5(f"{tool_name}:{arg_str}".encode("utf-8")).hexdigest()


async def _invoke_coro(coro, args, kwargs):
    """按调用形态（位置/关键字）执行底层 coroutine/func。"""
    call_args, call_kwargs = (args, kwargs) if args else ((), kwargs)
    if asyncio.iscoroutinefunction(coro):
        return await coro(*call_args, **call_kwargs)
    return coro(*call_args, **call_kwargs)


async def _cached_ainvoke(tool, original_coro, *args, **kwargs):
    if _AMAP_CACHE_DISABLED:
        return await _call_with_retry(original_coro, args, kwargs)
    # MCP 工具入参通常为单个 dict（StructuredTool 下以 **kwargs 传入）
    cache_arg = args[0] if args else kwargs
    key = _amap_cache_key(tool.name, cache_arg)
    now = time.time()
    hit = _AMAP_CACHE.get(key)
    if hit and now - hit[0] < _AMAP_CACHE_TTL:
        logger.info("[amap-cache] HIT %s", tool.name)
        return hit[1]
    result = await _call_with_retry(original_coro, args, kwargs)
    _AMAP_CACHE[key] = (now, result)
    logger.info("[amap-cache] MISS %s", tool.name)
    return result


def _wrap_tool_cache(tool):
    """给单个高德工具套一层 TTL 缓存。

    注意：langchain BaseTool 是 pydantic 模型，无法直接赋值 tool.ainvoke，
    因此改为返回一个「名称/描述/参数一致」的新 StructuredTool，底层 coroutine 走缓存。
    agent 内部调用与 hotel 直查调用均透明命中缓存。
    """
    schema = getattr(tool, "args_schema", None)
    original_coro = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
    if schema is None or original_coro is None:
        return tool  # 无法安全重建则跳过缓存包裹（降级为原工具）
    async def wrapped(*args, **kwargs):
        raw = await _cached_ainvoke(tool, original_coro, *args, **kwargs)
        # 对方向/地理编码类工具结果做摘要，避免把整段原始 JSON 回灌给模型
        if tool.name and ("direction" in tool.name.lower() or "geo" in tool.name.lower()):
            raw = _summarize_tool_output(tool.name, raw)
        return raw
    return StructuredTool.from_function(
        coroutine=wrapped,
        name=tool.name,
        description=tool.description or "",
        args_schema=schema,
    )

_client = None
_tools = None


def _build_client() -> MultiServerMCPClient:
    if not AMAP_API_KEY:
        raise RuntimeError("缺少 AMAP_API_KEY，请在 .env 中配置高德地图密钥")
    return MultiServerMCPClient(
        {
            "amap": {
                "url": f"https://mcp.amap.com/sse?key={AMAP_API_KEY}",
                "transport": "sse",
            }
        }
    )


async def get_amap_tools():
    """返回高德 MCP 工具列表（进程内单例）。失败返回空列表以降级。"""
    global _client, _tools
    if _tools is not None:
        return _tools

    if not AMAP_API_KEY:
        logger.warning("未配置 AMAP_API_KEY，地图工具不可用，将降级为纯模型生成。")
        _tools = []
        return _tools

    try:
        logger.info("正在初始化高德地图 MCP 连接...")
        _client = _build_client()
        _tools = await _client.get_tools()
        # 套一层工具级 TTL 缓存 + 结果压缩（单例只套一次，覆盖 agent 内部调用与直查调用）
        _tools = [_wrap_tool_cache(_t) for _t in _tools]
        logger.info("高德 MCP 工具加载完成，共 %d 个（已启用缓存+结果压缩）。", len(_tools))
    except Exception as e:  # noqa: BLE001
        logger.warning("高德 MCP 初始化失败，降级为纯模型生成：%s", e)
        _tools = []

    return _tools


def _iter_dicts(obj):
    """递归遍历嵌套结构，产出其中的 dict（用于定位 MCP content block）。"""
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for it in obj:
            yield from _iter_dicts(it)


def _extract_json(raw):
    """兼容高德 MCP 返回的多种形态，返回 dict/list，失败返回 None：
    - dict / list（已解析）
    - 纯 JSON 字符串
    - `('{json}', None)` 元组字符串（repr）
    - **MCP content-block 形态**：`([{'type':'text','text':'{json}'}], None)`

    第四种是此前 POI 解析与「方向/地理编码结果压缩」从未生效的根因：
    旧实现用正则取最外层 {…} 得到的是 Python 字面量（单引号），json.loads 必然失败。
    这里用 ast.literal_eval 安全还原外层结构（不执行代码），再取 content 的 text 解析。
    """
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        pass
    # 外层为元组/列表的 repr（含 MCP content block）→ 安全还原后解析 text 字段
    try:
        obj = ast.literal_eval(s)
    except Exception:  # noqa: BLE001
        obj = None
    if obj is not None:
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception:  # noqa: BLE001
                pass
        else:
            for d in _iter_dicts(obj):
                text = d.get("text")
                if isinstance(text, str):
                    try:
                        return json.loads(text)
                    except Exception:  # noqa: BLE001
                        return None
    # 退回：正则抽取最外层 JSON 对象
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None
    return None


def _payload(raw):
    """把高德 MCP 的返回归一为 dict/list：
    - content-block 形态（list/tuple 内含 {'type':'text','text':'{json}'}）→ 解析 text 里的 JSON；
    - 字符串形态 → 走 _extract_json；
    - 已是解析后的结构 → 原样返回。
    这是方向/地理编码压缩能生效的前提（此前因 list 被原样返回而失效）。
    """
    if isinstance(raw, (list, tuple)):
        for d in _iter_dicts(raw):
            text = d.get("text")
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:  # noqa: BLE001
                    return None
        # 元组/列表里直接放 JSON 字符串（如 ('{json}', None)）：解析首个字符串项。
        # 高德 MCP 的返回形态并不稳定，曾出现这种形态导致地理编码压缩失效、
        # 进而「坐标解析失败 → 酒店就近检索连锁回退」。
        for item in raw:
            if isinstance(item, str):
                parsed = _extract_json(item)
                if parsed is not None:
                    return parsed
        return raw  # 非 content-block（已解析结构）
    return _extract_json(raw)


def _parse_poi(raw, limit: int = 8) -> str:
    """把高德 POI 原始返回（JSON 字符串）压成「名称｜类型｜地址」精简列表，
    避免把整段原始 JSON 喂给 planner，既减小上下文，也便于阅读与后续结构化。"""
    if not raw:
        return ""
    data = _extract_json(raw)
    if data is None:
        return raw if isinstance(raw, str) else str(raw)
    pois = (data.get("pois") if isinstance(data, dict) else None) or []
    if not pois and isinstance(data, dict):
        pois = (data.get("suggestion") or {}).get("pois") or []
    if not pois:
        return raw if isinstance(raw, str) else str(raw)
    lines = [
        f"- {p.get('name', '')}｜{p.get('type', '') or ''}｜{p.get('address', '') or ''}"
        for p in pois[:limit]
    ]
    return "\n".join(lines)


def _truncate_str(s: str, max_chars: int = 2000) -> str:
    s = s if isinstance(s, str) else str(s)
    return s if len(s) <= max_chars else s[:max_chars] + "...(truncated)"


def _km(meters) -> str:
    try:
        return f"{float(meters) / 1000:.1f}km"
    except (TypeError, ValueError):
        return str(meters or "?")


def _min(seconds) -> str:
    try:
        s = float(seconds)
        if s < 60:
            return f"{int(s)}秒"
        return f"{int(s // 60)}分{int(s % 60)}秒"
    except (TypeError, ValueError):
        return str(seconds or "?")


def _summarize_geo(data) -> str:
    """地理编码返回压成「地址｜经纬度｜级别」精简行。兼容 geocodes / results 两种字段。"""
    geocodes = (data.get("geocodes") if isinstance(data, dict) else None) or (
        data.get("results") if isinstance(data, dict) else None
    ) or []
    if not geocodes:
        return _truncate_str(str(data))
    lines = []
    for g in geocodes[:3]:
        addr = (
            g.get("formatted_address")
            or f"{g.get('province', '')}{g.get('city', '')}{g.get('district', '')}{g.get('street', '')}".strip()
            or g.get("location", "")
        )
        level = g.get("level") or g.get("city") or g.get("name") or ""
        lines.append(f"{addr}｜{g.get('location', '')}｜{level}")
    return "\n".join(lines)


def _summarize_direction(data) -> str:
    """路线规划返回压成「总览 + 最快 2 方案概要」，丢弃逐段步行/转弯等冗余细节，
    减小回灌给模型的上下文（chat 最终回答 input 曾高达 7555 tok），也更易读。
    注意高德 MCP 返回与官方 API 不同：结果直接是顶层字段（无 `route` 包裹）。"""
    route = (data.get("route") if isinstance(data, dict) else None) or (data if isinstance(data, dict) else None)
    if not isinstance(route, dict):
        return _truncate_str(str(data))
    out = [f"起点：{route.get('origin', '')} 终点：{route.get('destination', '')}"]
    paths = route.get("paths")
    if paths:  # 驾车/步行/骑行
        for i, p in enumerate(paths[:2], 1):
            line = f"方案{i}｜{_km(p.get('distance'))} {_min(p.get('duration'))}"
            if p.get("tolls") not in (None, "", "0"):
                line += f" 过路费{p.get('tolls')}元"
            if p.get("traffic_lights") not in (None, "", "0"):
                line += f" 红绿灯{p.get('traffic_lights')}个"
            out.append(line)
        out.append("（逐段步行/转弯明细已省略）")
        return "\n".join(out)
    transits = route.get("transits")
    if transits:  # 公交/地铁：按耗时升序取最快 2 个，避免模型拿到步行占比高的慢方案
        def _dur(t):
            try:
                return int(t.get("duration") or 0)
            except (TypeError, ValueError):
                return 0
        ranked = sorted(transits, key=_dur)[:2]
        for i, t in enumerate(ranked, 1):
            cost = (t.get("cost") or {}).get("transit_fee") or (t.get("cost") or {}).get("price")
            dist = _km(t.get("distance"))
            line = f"方案{i}｜{_min(t.get('duration'))}" + (f" 全程{dist}" if dist != "?" else "")
            if cost not in (None, "", "0"):
                line += f" 票价{cost}元"
            segs = []
            for seg in t.get("segments", [])[:6]:
                bus = seg.get("bus") or {}
                if bus.get("buslines"):
                    bl = bus["buslines"][0]
                    segs.append(f"{bl.get('name', '')}({bl.get('departure_stop', {}).get('name', '')}→{bl.get('arrival_stop', {}).get('name', '')})")
                elif seg.get("walking", {}).get("steps"):
                    segs.append("步行")
                elif seg.get("railway"):
                    segs.append("火车/城际")
            if segs:
                line += "：" + " → ".join(segs)
            wd = t.get("walking_distance")
            if wd not in (None, "", "0"):
                line += f" ｜全程步行约{_km(wd)}"
            out.append(line)
        out.append("（逐段步行明细已省略；以上为耗时最短的方案）")
        return "\n".join(out)
    return _truncate_str(str(data))


def _summarize_tool_output(tool_name: str, raw) -> str:
    """对方向/地理编码类工具做结果压缩；其它工具原样返回（带降级截断）。"""
    name = (tool_name or "").lower()
    if "direction" not in name and "geo" not in name:
        return raw if isinstance(raw, str) else str(raw)
    data = _payload(raw)
    if data is None:
        return _truncate_str(raw if isinstance(raw, str) else str(raw), 2000)
    return _summarize_geo(data) if "geo" in name else _summarize_direction(data)


async def amap_poi_search(keywords: str, city: str = "") -> str:
    """确定性地调用高德 POI 搜索工具（不经由 LLM agent），返回「名称｜类型｜地址」精简文本。

    用于 hotel 等「只需真实参考数据、不需要 LLM 推理」的场景：直接拿地图 POI 结果，
    避免为简单检索多跑一个 LLM agent（点 4 / 点 7）。失败或工具不可用时降级为提示文本。
    """
    tools = await get_amap_tools()
    if not tools:
        return "（地图工具不可用，酒店信息暂缺）"
    # 优先匹配常见 POI 搜索工具名，其次退化为任意含 search 的工具
    search_tool = None
    for cand in ("maps_text_search", "maps_poi_search", "maps_poi", "maps_search"):
        search_tool = next((t for t in tools if t.name == cand), None)
        if search_tool:
            break
    if search_tool is None:
        search_tool = next((t for t in tools if "search" in t.name.lower()), None)
    if search_tool is None:
        return "（未找到 POI 搜索工具，酒店信息暂缺）"
    try:
        raw = await search_tool.ainvoke({"keywords": keywords, "city": city})
        text = raw if isinstance(raw, str) else str(raw)
        parsed = _parse_poi(text)
        return parsed or f"（未检索到「{keywords}」信息）"
    except Exception as e:  # noqa: BLE001
        logger.warning("高德 POI 搜索失败（%s）：%s", keywords, e)
        return f"（{keywords}查询失败：{e}；已降级）"


async def amap_weather(city: str) -> str:
    """确定性地调用高德天气工具（不经由 LLM agent），返回逐日预报精简文本。

    用于 weather 等确定性场景：高德 key 稳定可用，避免和风天气 key 绑定专属
    API Host / 配额受限时天气环节整体失败。失败返回空串，由调用方降级兜底。
    """
    tools = await get_amap_tools()
    if not tools:
        return ""
    weather_tool = next((t for t in tools if t.name == "maps_weather"), None)
    if weather_tool is None:
        weather_tool = next((t for t in tools if "weather" in t.name.lower()), None)
    if weather_tool is None:
        return ""
    try:
        raw = await weather_tool.ainvoke({"city": city})
        text = raw if isinstance(raw, str) else str(raw)
        return _format_amap_weather(city, text)
    except Exception as e:  # noqa: BLE001
        logger.warning("高德天气查询失败（%s）：%s", city, e)
        return ""


def _format_amap_weather(city: str, raw: str) -> str:
    """把 maps_weather 的 JSON（兼容元组字符串）压成逐日精简预报；无数据返回空串。"""
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return ""
    forecasts = data.get("forecasts") or []
    first = forecasts[0] if forecasts and isinstance(forecasts[0], dict) else {}
    casts = first.get("casts") or []
    if not casts:
        return ""
    lines = [f"【{first.get('city') or city} 天气预报】（来源：高德天气）"]
    for c in casts[:5]:
        lines.append(
            f"- {c.get('date')} 周{c.get('week')}：白天{c.get('dayweather')}/夜间{c.get('nightweather')}，"
            f"{c.get('nighttemp')}~{c.get('daytemp')}℃，{c.get('daywind')}风{c.get('daypower')}级"
        )
    return "\n".join(lines)
