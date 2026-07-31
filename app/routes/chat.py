import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)
from app.services.chat_service import ChatService
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Chat"])


def get_chat_service() -> ChatService:
    """
    Dependency that creates and initializes the chat service.
    """
    service = ChatService()
    service.initialize()
    return service


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatCompletionResponse:
    """
    OpenAI-compatible Chat Completions endpoint.
    """
    try:
        gemini_response = await chat_service.generate_response(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        content = gemini_response.text or ""

        usage_meta = getattr(gemini_response, "usage_metadata", None)

        prompt_tokens = (
            getattr(usage_meta, "prompt_token_count", 0)
            if usage_meta
            else 0
        )

        completion_tokens = (
            getattr(usage_meta, "candidates_token_count", 0)
            if usage_meta
            else 0
        )

        total_tokens = (
            getattr(usage_meta, "total_token_count", 0)
            if usage_meta
            else 0
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=content,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

    except Exception as e:
        logger.exception("Chat completion failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )