from pathlib import Path
from typing import Any, List, Optional

from google import genai
from google.genai import types
from google.genai.errors import ClientError

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
    """Raised when a chat request is invalid or the upstream Gemini service fails."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


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
    """Handle communication with Google Gemini using cached system instructions."""

    _client_cache: Optional[Any] = None

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        self.settings = get_settings()
        self.client: Any = None
        self.system_prompt = (
            system_prompt if system_prompt is not None else initialize_prompt_cache()
        )

    def initialize(self) -> Any:
        """Initialize the Gemini client once per process and reuse it across requests."""
        if self.client is not None:
            logger.debug("Gemini client already initialized; reusing existing client.")
            return self.client

        if not self.settings.GEMINI_API_KEY:
            raise ChatRequestError("Missing Gemini API key configuration.", status_code=500)

        try:
            if ChatService._client_cache is None:
                ChatService._client_cache = genai.Client(api_key=self.settings.GEMINI_API_KEY)
                logger.info(
                    "Gemini client initialized once for model %s",
                    self.settings.GEMINI_MODEL_NAME,
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to initialize Gemini client")
            raise ChatRequestError("Failed to initialize Gemini client.", status_code=500) from exc

        self.client = ChatService._client_cache
        return self.client

    def _build_system_instruction(self, messages: List[Message]) -> Optional[str]:
        """Combine the cached prompt with any client-supplied system messages."""
        has_system_messages = any(
            message.role == "system" and message.content.strip() for message in messages
        )
        if not self.system_prompt and not has_system_messages:
            return None

        instructions = [instruction for instruction in [self.system_prompt] if instruction]
        instructions.extend(
            message.content.strip()
            for message in messages
            if message.role == "system" and message.content.strip()
        )
        return "\n\n".join(instructions).strip() or None

    def _convert_messages_to_contents(self, messages: List[Message]) -> List[types.Content]:
        """Convert OpenAI-style messages into Gemini contents while preserving history order."""
        contents: List[types.Content] = []

        for message in messages:
            if message.role == "system":
                continue

            role = "model" if message.role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=message.content)],
                )
            )

        return contents

    def _validate_request(self, messages: List[Message], temperature: Optional[float]) -> None:
        """Validate the request payload before calling Gemini."""
        if not messages:
            raise ChatRequestError("At least one message is required.", status_code=400)

        if not any(message.content and message.content.strip() for message in messages):
            raise ChatRequestError("Messages must not be empty.", status_code=400)

        if temperature is None:
            return

        if not 0.0 <= temperature <= 2.0:
            raise ChatRequestError("Temperature must be between 0.0 and 2.0.", status_code=400)

    def _build_generation_config(
        self,
        temperature: float,
        max_tokens: Optional[int],
        system_instruction: Optional[str],
    ) -> types.GenerateContentConfig:
        """Build the Gemini generation config from the resolved request values."""
        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        return types.GenerateContentConfig(**config_kwargs)

    async def generate_response(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Generate a Gemini response for the supplied chat messages."""
        effective_temperature = (
            float(temperature)
            if temperature is not None
            else float(self.settings.GEMINI_TEMPERATURE)
        )
        self._validate_request(messages, effective_temperature)

        if self.client is None:
            raise ChatRequestError("Gemini client not initialized.", status_code=500)

        contents = self._convert_messages_to_contents(messages)
        system_instruction = self._build_system_instruction(messages)
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.settings.GEMINI_MAX_OUTPUT_TOKENS
        )
        config = self._build_generation_config(
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
            system_instruction=system_instruction,
        )

        logger.debug(
            "Sending %s message(s) to Gemini with system instruction enabled.",
            len(contents),
        )

        try:
            return self.client.models.generate_content(
                model=self.settings.GEMINI_MODEL_NAME,
                contents=contents,
                config=config,
            )
        except ClientError as exc:
            logger.exception("Gemini API request failed for model %s", self.settings.GEMINI_MODEL_NAME)
            raise ChatRequestError("Gemini API request failed.", status_code=502) from exc
        except Exception as exc:
            logger.exception("Unexpected Gemini generation failure for model %s", self.settings.GEMINI_MODEL_NAME)
            raise ChatRequestError("Gemini request failed.", status_code=502) from exc
