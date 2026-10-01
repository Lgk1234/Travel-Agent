"""配置加载：从 .env 读取各厂商 API Key / URL 与默认模型。"""
import os

from dotenv import load_dotenv

load_dotenv(override=True)

# 默认模型（统一走百炼，仅改模型名即可切换）
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")

# 大模型统一网关（阿里云百炼 OpenAI 兼容接口）
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")

# 高德地图 MCP（必填）
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

# 和风天气（自定义 tool 接入，GeoAPI）
QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")
QWEATHER_API_HOST = os.getenv("QWEATHER_API_HOST")

# LangSmith（可选）
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")

if LANGSMITH_API_KEY and LANGSMITH_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "smart-travel-agent")
