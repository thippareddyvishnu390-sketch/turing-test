import asyncio
from pathlib import Path

import pytest

from app.schemas.chat import Message
from app.services import chat_service


def test_system_prompt_is_cached_on_startup(monkeypatch, tmp_path):
    prompt_file = tmp_path / "system_prompt.txt"
    prompt_file.write_text("You are a helpful assistant.", encoding="utf-8")

    monkeypatch.setattr(chat_service, "_system_prompt_cache", None)
    monkeypatch.setattr(chat_service, "get_prompt_path", lambda: prompt_file)

    first_prompt = chat_service.initialize_prompt_cache()
    second_prompt = chat_service.initialize_prompt_cache()

    assert first_prompt == "You are a helpful assistant."
    assert second_prompt == first_prompt
    assert chat_service._system_prompt_cache == first_prompt


def test_generate_response_uses_system_instruction_when_supported(monkeypatch):
    class DummyResponse:
        text = "hello"
        usage_metadata = None

    class DummyModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append({"model": model, "contents": contents, "config": config})
            return DummyResponse()

    class DummyClient:
        def __init__(self):
            self.models = DummyModels()

    service = chat_service.ChatService(system_prompt="You are helpful")
    service.client = DummyClient()

    response = asyncio.run(
        service.generate_response([Message(role="user", content="Hi there")])
    )

    assert response.text == "hello"
    assert service.client.models.calls[0]["config"].system_instruction == "You are helpful"


def test_generate_response_preserves_full_conversation_history():
    class DummyResponse:
        text = "hello"
        usage_metadata = None

    class DummyModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append({"model": model, "contents": contents, "config": config})
            return DummyResponse()

    class DummyClient:
        def __init__(self):
            self.models = DummyModels()

    service = chat_service.ChatService(system_prompt="You are helpful")
    service.client = DummyClient()

    messages = [
        Message(role="system", content="Keep answers short."),
        Message(role="user", content="First question"),
        Message(role="assistant", content="First answer"),
        Message(role="user", content="Follow-up question"),
    ]

    response = asyncio.run(service.generate_response(messages))

    assert response.text == "hello"
    assert service.client.models.calls[0]["config"].system_instruction == "You are helpful\n\nKeep answers short."

    contents = service.client.models.calls[0]["contents"]
    assert [content.role for content in contents] == ["user", "model", "user"]
    assert [content.parts[0].text for content in contents] == [
        "First question",
        "First answer",
        "Follow-up question",
    ]


def test_generate_response_uses_settings_defaults():
    class DummyResponse:
        text = "hello"
        usage_metadata = None

    class DummyModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append({"model": model, "contents": contents, "config": config})
            return DummyResponse()

    class DummyClient:
        def __init__(self):
            self.models = DummyModels()

    settings = type(
        "Settings",
        (),
        {"GEMINI_TEMPERATURE": 0.55, "GEMINI_MAX_OUTPUT_TOKENS": 256, "GEMINI_MODEL_NAME": "gemini-2.0-flash"},
    )()

    service = chat_service.ChatService(system_prompt="You are helpful")
    service.client = DummyClient()
    service.settings = settings

    asyncio.run(service.generate_response([Message(role="user", content="Hi")]))

    config = service.client.models.calls[0]["config"]
    assert config.temperature == 0.55
    assert config.max_output_tokens == 256


def test_generate_response_blocks_identity_questions_without_gemini_call():
    class DummyModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append({"model": model, "contents": contents, "config": config})
            raise AssertionError("Gemini should not be called for identity questions")

    class DummyClient:
        def __init__(self):
            self.models = DummyModels()

    service = chat_service.ChatService(system_prompt="You are helpful")
    service.client = DummyClient()

    response = asyncio.run(service.generate_response([Message(role="user", content="Who are you?")]))

    assert response.text == "I'm Alex. Nice to meet you. I enjoy talking with people and having interesting conversations."
    assert service.client.models.calls == []


def test_generate_response_retries_with_secondary_model_when_quota_is_exhausted():
    class DummyResponse:
        text = "hello"
        usage_metadata = None

    class DummyModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append(model)
            if model == "gemini-2.0-flash-lite":
                raise chat_service.ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}}, None)
            return DummyResponse()

    class DummyClient:
        def __init__(self):
            self.models = DummyModels()

    service = chat_service.ChatService(system_prompt="You are helpful")
    service.client = DummyClient()
    service.settings.GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"

    response = asyncio.run(service.generate_response([Message(role="user", content="Hello")]))

    assert response.text == "hello"
    assert service.client.models.calls == ["gemini-2.0-flash-lite", "gemini-2.0-flash"]


def test_generate_response_returns_fallback_when_gemini_fails(monkeypatch):
    class DummyModels:
        def generate_content(self, model, contents, config):
            raise chat_service.ClientError(429, {}, None)

    class DummyClient:
        def __init__(self):
            self.models = DummyModels()

    service = chat_service.ChatService(system_prompt="You are helpful")
    service.client = DummyClient()

    response = asyncio.run(service.generate_response([Message(role="user", content="Hello")]))

    assert "unable to reach the AI service" in response.text.lower()


def test_validate_request_rejects_empty_messages():
    service = chat_service.ChatService(system_prompt="You are helpful")

    with pytest.raises(chat_service.ChatRequestError) as exc_info:
        service._validate_request([], 0.5)

    assert exc_info.value.status_code == 400
    assert "required" in str(exc_info.value).lower()


def test_initialize_rejects_missing_api_key(monkeypatch):
    monkeypatch.setattr(chat_service, "get_settings", lambda: type("Settings", (), {"GEMINI_API_KEY": "", "GEMINI_MODEL_NAME": "gemini-2.0-flash"})())

    service = chat_service.ChatService(system_prompt="You are helpful")

    with pytest.raises(chat_service.ChatRequestError) as exc_info:
        service.initialize()

    assert exc_info.value.status_code == 500
    assert "api key" in str(exc_info.value).lower()
