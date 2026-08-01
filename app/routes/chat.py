import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)
from app.services.chat_service import ChatRequestError, ChatService
from app.utils.logging import (
    get_logger,
    log_api_failure,
    log_request_completed,
    time_request,
)

logger = get_logger(__name__)

router = APIRouter(tags=["Chat"])


def get_chat_service(request: Request) -> ChatService:
    """Retrieve the shared chat service from application state."""
    service: ChatService | None = getattr(request.app.state, "chat_service", None)

    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chat service is not available.")

    return service


def _build_usage_payload(gemini_response: Any) -> Usage:
    """Extract token usage from the Gemini response into the public schema."""
    usage_meta = getattr(gemini_response, "usage_metadata", None)

    return Usage(
        prompt_tokens=getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0,
        completion_tokens=getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0,
        total_tokens=getattr(usage_meta, "total_token_count", 0) if usage_meta else 0,
    )


def _build_response_payload(request: ChatCompletionRequest, content: str, usage: Usage) -> ChatCompletionResponse:
    """Build the OpenAI-compatible response model from the Gemini result."""
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatCompletionResponse:
    """Handle an OpenAI-compatible chat completion request."""
    request_id = f"chat-{uuid.uuid4()}"
    started_at = time_request(
        logger,
        request_id,
        method="POST",
        path="/chat/completions",
        model=request.model,
        message_count=len(request.messages),
    )

    try:
        gemini_response = await chat_service.generate_response(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        content = gemini_response.text or ""
        usage = _build_usage_payload(gemini_response)

        duration_ms = (time.perf_counter() - started_at) * 1000
        log_request_completed(
            logger,
            request_id,
            duration_ms,
            model=request.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

        return _build_response_payload(request, content, usage)

    except ChatRequestError as exc:
        duration_ms = (time.perf_counter() - started_at) * 1000
        log_api_failure(
            logger,
            request_id,
            "chat request rejected",
            model=request.model,
            duration_ms=duration_ms,
            error=str(exc),
        )
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        duration_ms = (time.perf_counter() - started_at) * 1000
        log_api_failure(
            logger,
            request_id,
            "chat completion failed",
            model=request.model,
            duration_ms=duration_ms,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat completion failed.",
        ) from exc