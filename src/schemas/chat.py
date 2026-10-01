"""对话问答相关的数据模型。"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户当前输入的消息")
    history: List[ChatMessage] = Field(
        default_factory=list, description="历史对话（用于多轮上下文）"
    )
    model: Optional[str] = Field(
        default=None, description="可选：临时覆盖默认模型（如 deepseek-chat / gpt-4o）"
    )
    thread_id: Optional[str] = Field(
        default=None, description="会话线程 ID，用于短期记忆（LangGraph checkpoint）"
    )
