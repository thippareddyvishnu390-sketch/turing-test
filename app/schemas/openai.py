from typing import List, Optional, Any, Literal
from pydantic import BaseModel, Field


class OpenAIError(BaseModel):
    """
    Standard OpenAI error object format.
    """
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    """
    Top-level OpenAI error response wrapper.
    """
    error: OpenAIError


class OpenAIUsage(BaseModel):
    """
    Usage statistics for the completion request.
    """
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt.")
    completion_tokens: int = Field(..., description="Number of tokens in the generated completion.")
    total_tokens: int = Field(..., description="Total number of tokens used in the request.")


class ChatCompletionDelta(BaseModel):
    """
    Delta object for streaming responses.
    """
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """
    Choice object for streaming chunks.
    """
    index: int
    delta: ChatCompletionDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None


class ChatCompletionChunk(BaseModel):
    """
    Represents a chunk of a streaming chat completion response.
    """
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]