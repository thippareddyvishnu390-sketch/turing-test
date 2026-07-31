from pathlib import Path
from typing import List

from google import genai
from google.genai import types

from app.config.settings import get_settings
from app.schemas.chat import Message
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ChatService:
    """
    Handles communication with Google Gemini.
    Loads the system prompt once and sends it with every request.
    """

    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self.system_prompt = self.load_system_prompt()

    def load_system_prompt(self) -> str:
        """
        Load the system prompt from app/prompts/system_prompt.txt
        """

        try:
            prompt_path = (
                Path(__file__).parent.parent
                / "prompts"
                / "system_prompt.txt"
            )

            if prompt_path.exists():
                logger.info("System prompt loaded successfully.")
                return prompt_path.read_text(
                    encoding="utf-8"
                ).strip()

            logger.warning("system_prompt.txt not found.")
            return ""

        except Exception as e:
            logger.error(f"Failed to load system prompt: {e}")
            return ""

    def initialize(self):
        """
        Initialize Gemini client.
        """

        self.client = genai.Client(
            api_key=self.settings.GEMINI_API_KEY
        )

        logger.info("Gemini initialized.")
        logger.info(
            f"Using model: {self.settings.GEMINI_MODEL_NAME}"
        )

    async def generate_response(
        self,
        messages: List[Message],
        temperature: float = 0.8,
        max_tokens: int | None = None,
    ):

        if self.client is None:
            raise ValueError("Gemini client not initialized.")

        contents = []

        # Add system prompt first
        if self.system_prompt:

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=self.system_prompt
                        )
                    ],
                )
            )

        # Add conversation history
        for msg in messages:

            role = (
                "user"
                if msg.role == "user"
                else "model"
            )

            contents.append(
                types.Content(
                    role=role,
                    parts=[
                        types.Part.from_text(
                            text=msg.content
                        )
                    ],
                )
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = self.client.models.generate_content(
            model=self.settings.GEMINI_MODEL_NAME,
            contents=contents,
            config=config,
        )

        return response