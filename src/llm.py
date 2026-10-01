"""大模型统一接入层。

- 模块级 `llm`：默认模型实例，直接 `from src.llm import llm` 即可用。
- `get_llm(model)`：按名称返回实例，业务层零耦合具体厂商。
所有模型统一经阿里云百炼（Bailian）OpenAI 兼容网关接入（ChatOpenAI），
仅需通过 model 名切换，不在代码里区分厂商。
"""
from langchain_openai import ChatOpenAI

from src.config import (
    DEFAULT_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
)


def _build(model: str, disable_thinking: bool = False) -> ChatOpenAI:
    kwargs = dict(
        model=model,
        api_key=LLM_API_KEY or "EMPTY",
        base_url=LLM_BASE_URL,
        streaming=True,
        timeout=60,
        max_retries=2,
    )
    if disable_thinking:
        # 对话页统一关闭思考模式（不区分具体模型）。qwen3 等混合思考模型通过
        # extra_body.enable_thinking=false 关闭，显著降低首 token 延迟与输出 token 数。
        # 注意：ChatOpenAI 支持将 extra_body 作为顶层参数直接传入（避免 model_kwargs 警告）。
        kwargs["extra_body"] = {"enable_thinking": False}
    return ChatOpenAI(**kwargs)


def get_llm(model: str | None = None, temperature: float = 0.5, disable_thinking: bool = False) -> ChatOpenAI:
    """按模型名返回 ChatOpenAI 实例（统一走阿里云百炼网关，模型名直接透传）。

    - 无需区分厂商：所有百炼模型共用同一 API key / base_url，仅 model 名不同。
    - disable_thinking：关闭思考模式（用于对话页，降低延迟）。
    """
    return _build(model or DEFAULT_MODEL, disable_thinking=disable_thinking)


# 默认模型实例，供直接引用
llm = get_llm()
