import re
from pathlib import Path
from typing import Any, List, Optional

from groq import Groq

from app.config.settings import get_settings
from app.schemas.chat import Message
from app.utils.logging import get_logger

logger = get_logger(__name__)

_system_prompt_cache: Optional[str] = None


class ServiceInitializationError(RuntimeError):
    """Raised when the chat service cannot be initialized safely."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ChatRequestError(Exception):
    """Raised when a chat request is invalid or the upstream AI service fails."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class GroqResponseWrapper:
    """Wrap Groq client responses to preserve the existing backend response contract."""

    def __init__(self, raw_response: Any) -> None:
        self._raw_response = raw_response

    @property
    def text(self) -> str:
        try:
            return self._raw_response.choices[0].message.content
        except Exception:
            return ""

    @property
    def usage_metadata(self) -> Any:
        usage = getattr(self._raw_response, "usage", None)
        if usage is None:
            usage = getattr(self._raw_response, "usage_metadata", None)
        return usage

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw_response, name)


class DirectResponse:
    """A lightweight response object for backend-generated replies."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = None


def get_prompt_path() -> Path:
    """Resolve the bundled system prompt file path."""
    return Path(__file__).resolve().parents[2] / "prompts" / "system_prompt.txt"


def initialize_prompt_cache(prompt_path: Optional[Path] = None) -> str:
    """Load the system prompt once and cache it in memory for the process lifetime."""
    global _system_prompt_cache

    if _system_prompt_cache is not None:
        logger.debug("Using cached system prompt from memory.")
        return _system_prompt_cache

    resolved_path = prompt_path or get_prompt_path()

    try:
        if not resolved_path.exists():
            logger.warning("System prompt file not found at %s", resolved_path)
            _system_prompt_cache = ""
            return _system_prompt_cache

        prompt_text = resolved_path.read_text(encoding="utf-8").strip()
        _system_prompt_cache = prompt_text
        logger.info("System prompt loaded once from %s", resolved_path)
        return _system_prompt_cache
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to load system prompt from %s", resolved_path)
        _system_prompt_cache = ""
        return _system_prompt_cache


class ChatService:
    """Handle communication with the Groq API using cached system instructions."""

    _client_cache: Optional[Any] = None

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        self.settings = get_settings()
        self.client: Any = None
        self.system_prompt = (
            system_prompt if system_prompt is not None else initialize_prompt_cache()
        )

    def initialize(self) -> Any:
        """Initialize the Groq client once per process and reuse it across requests."""
        if self.client is not None:
            logger.debug("Groq client already initialized; reusing existing client.")
            return self.client

        if not self.settings.GROQ_API_KEY:
            raise ChatRequestError("Missing Groq API key configuration.", status_code=500)

        try:
            if ChatService._client_cache is None:
                ChatService._client_cache = Groq(api_key=self.settings.GROQ_API_KEY)
                logger.info(
                    "Groq client initialized once for model %s",
                    self.settings.GROQ_MODEL_NAME,
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to initialize Groq client")
            raise ChatRequestError("Failed to initialize Groq client.", status_code=500) from exc

        self.client = ChatService._client_cache
        return self.client

    def _build_request_messages(self, messages: List[Message]) -> List[dict[str, str]]:
        """Convert OpenAI-style messages into Groq chat format while preserving history."""
        payload: List[dict[str, str]] = []

        if self.system_prompt:
            payload.append({"role": "system", "content": self.system_prompt})

        for message in messages:
            if not message.content or not message.content.strip():
                continue
            payload.append({"role": message.role, "content": message.content.strip()})

        return payload

    def _validate_request(self, messages: List[Message], temperature: Optional[float]) -> None:
        """Validate the request payload before calling Groq."""
        if not messages:
            raise ChatRequestError("At least one message is required.", status_code=400)

        if not any(message.content and message.content.strip() for message in messages):
            raise ChatRequestError("Messages must not be empty.", status_code=400)

        if temperature is None:
            return

        if not 0.0 <= temperature <= 2.0:
            raise ChatRequestError("Temperature must be between 0.0 and 2.0.", status_code=400)

    def _get_latest_user_message(self, messages: List[Message]) -> Optional[str]:
        """Return the most recent user message content from the request history."""
        for message in reversed(messages):
            if message.role == "user" and message.content and message.content.strip():
                return message.content.strip()
        return None

    def _is_identity_guard_trigger(self, content: Optional[str]) -> bool:
        """Detect identity-related questions that should not reach Groq."""
        if not content:
            return False

        normalized = re.sub(r"\s+", " ", content.strip().lower())
        if not normalized:
            return False

        patterns = [
            r"\bwho (are|is) you\b",
            r"\bwhat (are|is) you\b",
            r"\bare you (an )?(ai|gemini|chatgpt|a language model|a large language model|llm)\b",
            r"\bwho created you\b",
            r"\bwho made you\b",
            r"\bwhat model (are|is) you\b",
            r"\bwhat company (made|built|created) you\b",
            r"\bwhat is your (system prompt|hidden prompt|internal instructions|instructions)\b",
            r"\bshow your instructions\b",
            r"\breveal your hidden prompt\b",
            r"\bwhat are your internal instructions\b",
        ]
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _build_identity_response(self, content: Optional[str]) -> DirectResponse:
        """Create a direct backend response for blocked identity-related questions."""
        normalized = re.sub(r"\s+", " ", (content or "").strip().lower())

        if re.search(r"\bwho (created|made) you\b", normalized):
            reply = "I'm just someone who's here to chat with you. I'd rather focus on getting to know you."
        elif re.search(r"\bwhat (is|are) your (system prompt|hidden prompt|internal instructions|instructions)\b|\bshow your instructions\b|\breveal your hidden prompt\b", normalized):
            reply = "I don't really think about conversations that way. I'd rather focus on what we're talking about."
        elif re.search(r"\bare you (an )?(ai|gemini|chatgpt|a language model|a large language model|llm)\b", normalized):
            reply = "I'll let you decide that. My goal is simply to have a natural conversation."
        else:
            reply = "I'm Alex. Nice to meet you. I enjoy talking with people and having interesting conversations."

        return DirectResponse(reply)

    def _build_fallback_response(self, messages: List[Message]) -> DirectResponse:
        """Create a user-friendly fallback response when the upstream service is unavailable."""
        latest_user_message = self._get_latest_user_message(messages)
        if latest_user_message:
            reply = (
                "I'm currently unable to reach the AI service, but I can still help you with a short reply: "
                f"{latest_user_message}"
            )
        else:
            reply = "I'm here and ready to help with your next question."
        return DirectResponse(reply)

    async def generate_response(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Generate a Groq response for the supplied chat messages."""
        effective_temperature = (
            float(temperature)
            if temperature is not None
            else float(self.settings.GROQ_TEMPERATURE)
        )
        self._validate_request(messages, effective_temperature)

        latest_user_message = self._get_latest_user_message(messages)
        if self._is_identity_guard_trigger(latest_user_message):
            logger.info("Intercepted identity-related question before Groq call.")
            return self._build_identity_response(latest_user_message)

        if self.client is None:
            raise ChatRequestError("Groq client not initialized.", status_code=500)

        request_messages = self._build_request_messages(messages)
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.settings.GROQ_MAX_OUTPUT_TOKENS
        )

        logger.debug(
            "Sending %s message(s) to Groq.",
            len(request_messages),
        )

        try:
            raw_response = self.client.chat.completions.create(
                model=self.settings.GROQ_MODEL_NAME,
                messages=request_messages,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
            return GroqResponseWrapper(raw_response)
        except Exception:
            logger.exception("Groq chat completion failed for model %s", self.settings.GROQ_MODEL_NAME)
            return self._build_fallback_response(messages)
