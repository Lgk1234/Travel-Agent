"""和风天气自定义工具（@tool 封装）：供 Chat Agent 与 Planning 的 Weather Node 调用。

按和风开发指南做职责分离：
- **GeoAPI**（`/geo/v2/city/lookup`）：只负责地理定位 —— 城市名 → 经纬度。
- **Weather API**（`/v7/weather/now`、`/v7/weather/{3,7}d`、`/v7/indices/1d`）：
  只负责天气查询 —— 经纬度 → 实况 / 逐日预报 / 生活指数。
- 两者统一使用和风控制台生成的**独立 API Host**，通过 `X-QW-Api-Key` 请求头鉴权。

整体流程：
    城市 → GeoAPI 解析经纬度 → Weather API 查询天气 → 格式化 JSON → 返回给 Agent
    由 LLM 结合上下文生成自然语言回复；Agent 层不直接调 HTTP 接口，只用本 @tool。

稳定性：
- 城市解析与天气查询均带缓存，避免重复请求；
- 任一环节失败只降级为明确文本（不抛异常），不影响整体流程；
- 绝不编造温度/天气等数值。
"""
import logging
import os
import time
from datetime import date, datetime
from typing import List, Optional, Tuple

import httpx
from langchain_core.tools import tool

from src.config import QWEATHER_API_KEY, QWEATHER_API_HOST

logger = logging.getLogger("weather_tool")

# 和风控制台生成的独立 API Host：Geo 与 Weather 共用同一 Host，仅路径不同
API_HOST = (QWEATHER_API_HOST or "https://api.qweather.com").strip().rstrip("/")

# 路径：GeoAPI（定位）与 Weather API（天气）职责分离
GEO_PATH = "/geo/v2/city/lookup"
WEATHER_NOW_PATH = "/v7/weather/now"
WEATHER_DAILY_PATH = "/v7/weather/{days}d"  # days ∈ {3, 7, 10, 15, 30}
INDICES_PATH = "/v7/indices/1d"

# 缓存：避免重复请求（城市解析长期缓存；天气结果 TTL 缓存）
_GEO_CACHE: dict[str, tuple] = {}
_WEATHER_CACHE: dict[str, tuple[float, str]] = {}
_WEATHER_CACHE_TTL = float(os.getenv("QWEATHER_CACHE_TTL", "3600"))  # 默认 1 小时
_CACHE_DISABLED = os.getenv("DISABLE_QWEATHER_CACHE", "").lower() in ("1", "true", "yes")


def _headers() -> dict:
    """和风鉴权：X-QW-Api-Key 请求头。"""
    return {"X-QW-Api-Key": QWEATHER_API_KEY or ""}


async def _get_json(path: str, params: dict) -> Optional[dict]:
    """统一 GET：失败只告警并返回 None（降级），不把异常抛给调用方。"""
    if not QWEATHER_API_KEY:
        return None
    url = f"{API_HOST}{path}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params, headers=_headers())
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("和风天气请求失败 %s: %s", url, e)
        return None
    # 和风以 HTTP 200 + code 表达业务状态，非 "200" 视为失败
    if isinstance(data, dict) and data.get("code") not in (None, "200"):
        logger.warning("和风天气返回业务错误 %s: code=%s", url, data.get("code"))
        return None
    return data


async def geocode(city: str) -> Optional[tuple]:
    """GeoAPI：城市名 → (lat, lon)。仅做地理定位，带缓存。"""
    if not city:
        return None
    if city in _GEO_CACHE:
        return _GEO_CACHE[city]
    data = await _get_json(GEO_PATH, {"location": city, "number": 1})
    loc = (data or {}).get("location") or []
    if not loc:
        logger.warning("GeoAPI 未解析到城市「%s」的坐标", city)
        return None
    item = loc[0]
    coord = (item.get("lat"), item.get("lon"))
    _GEO_CACHE[city] = coord
    return coord


def _needed_daily_days(days: int, end_date: str) -> int:
    """根据行程结束日期决定请求 3d 还是 7d（v7 逐日仅支持 3d/7d/10d/15d/30d）。"""
    if end_date:
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
            if (end - date.today()).days >= 3:
                return 7
        except ValueError:
            pass
    return 3 if days <= 3 else 7


def _select_casts(casts: List[dict], days: int, start_date: str,
                  end_date: str) -> Tuple[List[dict], str]:
    """按行程日期区间挑选逐日预报；行程超出预报范围时如实说明（不编造）。"""
    if start_date and end_date:
        selected = [c for c in casts if start_date <= (c.get("fxDate") or "") <= end_date]
        if selected:
            return selected, ""
        first = (casts[0].get("fxDate") if casts else "")
        return casts[:days], (
            f"（注意：当前预报仅覆盖 {first} 起 {len(casts)} 天，未覆盖行程日期 "
            f"{start_date}~{end_date}，请在出行前再次查询官方预报）"
        )
    return casts[:days], ""


def _format(city: str, now: Optional[dict], daily: Optional[dict],
            indices: Optional[dict], days: int,
            start_date: str = "", end_date: str = "") -> str:
    """把天气 JSON 格式化为给 Agent 的文本（不编造数值）。"""
    lines = [f"【{city} 天气预报】（来源：和风天气）"]

    now_data = (now or {}).get("now") or {}
    if now_data:
        lines.append(
            f"实况：{now_data.get('text')} {now_data.get('temp')}℃ "
            f"（体感 {now_data.get('feelsLike')}℃，"
            f"{now_data.get('windDir')}{now_data.get('windScale')}级，"
            f"湿度 {now_data.get('humidity')}%）"
        )

    casts = (daily or {}).get("daily") or []
    selected, note = _select_casts(casts, days, start_date, end_date)
    if selected:
        title = f"逐日预报（行程 {start_date} ~ {end_date}）：" if (start_date and end_date) else "逐日预报："
        lines.append(title)
        for d in selected:
            lines.append(
                f"- {d.get('fxDate')}：{d.get('textDay')} "
                f"{d.get('tempMin')}~{d.get('tempMax')}℃ "
                f"（{d.get('windDirDay')}{d.get('windScaleDay')}级）"
            )
    if note:
        lines.append(note)

    idx_list = (indices or {}).get("daily") or []
    if idx_list:
        lines.append("生活指数：")
        for it in idx_list:
            lines.append(f"- {it.get('name')}：{it.get('category')}（{it.get('text')}）")

    if len(lines) == 1:
        return f"【天气工具】{city} 天气数据暂不可用，请稍后重试或改用其他来源。"
    return "\n".join(lines)


async def fetch_weather(city: str, days: int = 3,
                        start_date: str = "", end_date: str = "") -> str:
    """完整天气查询：GeoAPI 定位 → Weather API 查询 → 按行程日期格式化返回文本。

    带缓存；任一环节失败返回明确降级文本，不抛异常。
    start_date/end_date 提供时，只返回该区间内的逐日预报。
    """
    if not QWEATHER_API_KEY:
        return "【天气工具不可用】未配置 QWEATHER_API_KEY，请配置和风天气密钥。"

    days = max(1, min(7, days))
    cache_key = f"{city}:{days}:{start_date}:{end_date}"
    if not _CACHE_DISABLED:
        hit = _WEATHER_CACHE.get(cache_key)
        if hit and time.time() - hit[0] < _WEATHER_CACHE_TTL:
            logger.info("[weather-cache] HIT %s", cache_key)
            return hit[1]

    # 1) 地理定位：城市 → 经纬度（GeoAPI）
    coord = await geocode(city)
    if not coord:
        return f"【天气工具】未找到城市「{city}」的坐标，请检查城市名。"
    lat, lon = coord
    location = f"{lon},{lat}"  # Weather API 位置参数：经度,纬度

    # 2) 天气查询：经纬度 → 实况 / 逐日 / 生活指数（Weather API）
    daily_days = _needed_daily_days(days, end_date)  # 按行程结束日期选 3d/7d
    now = await _get_json(WEATHER_NOW_PATH, {"location": location})
    daily = await _get_json(WEATHER_DAILY_PATH.format(days=daily_days), {"location": location})
    indices = await _get_json(INDICES_PATH, {"location": location, "type": "1,3,5,8,9"})

    # 和风整体不可用 → 降级到高德天气（不编造数据）
    if now is None and daily is None and indices is None:
        return await _amap_fallback(city, cache_key)

    # 3) 格式化后返回给 Agent
    text = _format(city, now, daily, indices, days, start_date, end_date)
    if not _CACHE_DISABLED:
        _WEATHER_CACHE[cache_key] = (time.time(), text)
    return text


async def _amap_fallback(city: str, cache_key: str) -> str:
    """和风不可用时回退高德天气；两者都失败则返回明确降级文本。"""
    try:
        from src.tools.amap.client import amap_weather  # 延迟导入，避免与 amap 侧循环依赖

        text = await amap_weather(city)
    except Exception as e:  # noqa: BLE001
        logger.warning("高德天气回退失败：%s", e)
        text = ""
    if text:
        if not _CACHE_DISABLED:
            _WEATHER_CACHE[cache_key] = (time.time(), text)
        return text
    return f"【天气工具】{city} 天气数据暂不可用（和风与高德来源均失败）。"


@tool
async def weather_tool(city: str, days: int = 3,
                       start_date: str = "", end_date: str = "") -> str:
    """查询指定城市的天气：实况、未来逐日预报与生活指数（穿衣/紫外线/舒适度等）。

    当用户询问天气，或行程规划需要天气依据时使用。
    参数:
      city: 中文城市名，如「北京」「上海」
      days: 预报天数，1-7，默认 3
      start_date/end_date: 行程起止日期（YYYY-MM-DD），提供时只返回该区间内的预报
    """
    return await fetch_weather(city, days, start_date, end_date)
