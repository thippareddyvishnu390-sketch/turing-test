from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field


class Message(BaseModel):
    """
    OpenAI-compatible message schema.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """
    Schema for OpenAI Chat Completion requests.
    """
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=128)
    stream: Optional[bool] = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None


class Choice(BaseModel):
    """
    Schema for a single completion choice in the response.
    """
    index: int
    message: Message
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "null"]] = "stop"


class Usage(BaseModel):
    """
    Schema for token usage statistics.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """
    Schema for OpenAI-compatible Chat Completion responses.
    """
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage