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


def test_generate_response_uses_system_prompt_when_supported():
    class DummyResponse:
        # Groq wrapper expects .choices[0].message.content; mimic raw structure
        class Choice:
            class Message:
                def __init__(self, content):
                    self.content = content

            def __init__(self, content):
                self.message = DummyResponse.Choice.Message(content)

        def __init__(self):
            self.choices = [DummyResponse.Choice("hello")]
            self.usage = None

    class DummyCompletions:
        def __init__(self):
            self.calls = []

        def create(self, model, messages, temperature, max_tokens):
            self.calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            return DummyResponse()

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self):
            self.chat = DummyChat()

    service = chat_service.ChatService(system_prompt="You are helpful")
    service.client = DummyClient()

    response = asyncio.run(
        service.generate_response([Message(role="user", content="Hi there")])
    )

    assert response.text == "hello"
    assert service.client.chat.completions.calls[0]["model"] == service.settings.GROQ_MODEL_NAME
    assert service.client.chat.completions.calls[0]["messages"] == [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hi there"},
    ]


def test_generate_response_preserves_full_conversation_history():
    class DummyResponse:
        class Choice:
            class Message:
                def __init__(self, content):
                    self.content = content

            def __init__(self, content):
                self.message = DummyResponse.Choice.Message(content)

        def __init__(self):
            self.choices = [DummyResponse.Choice("hello")]
            self.usage = None

    class DummyCompletions:
        def __init__(self):
            self.calls = []

        def create(self, model, messages, temperature, max_tokens):
            self.calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            return DummyResponse()

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self):
            self.chat = DummyChat()

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

    call = service.client.chat.completions.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "You are helpful"},
        {"role": "system", "content": "Keep answers short."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow-up question"},
    ]


def test_generate_response_uses_settings_defaults():
    class DummyResponse:
        class Choice:
            class Message:
                def __init__(self, content):
                    self.content = content

            def __init__(self, content):
                self.message = DummyResponse.Choice.Message(content)

        def __init__(self):
            self.choices = [DummyResponse.Choice("hello")]
            self.usage = None

    class DummyCompletions:
        def __init__(self):
            self.calls = []

        def create(self, model, messages, temperature, max_tokens):
            self.calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            return DummyResponse()

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self):
            self.chat = DummyChat()

    settings = type(
        "Settings",
        (),
        {
            "GROQ_TEMPERATURE": 0.55,
            "GROQ_MAX_OUTPUT_TOKENS": 256,
            "GROQ_MODEL_NAME": "llama-3.3-70b-versatile",
        },
    )()

    service = chat_service.ChatService(system_prompt="You are helpful")
    service.client = DummyClient()
    service.settings = settings

    asyncio.run(service.generate_response([Message(role="user", content="Hi")]))

    call = service.client.chat.completions.calls[0]
    assert call["temperature"] == 0.55
    assert call["max_tokens"] == 256


def test_generate_response_blocks_identity_questions_without_groq_call():
    class DummyCompletions:
        def __init__(self):
            self.calls = []

        def create(self, model, messages, temperature, max_tokens):
            self.calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            raise AssertionError("Groq should not be called for identity questions")

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self):
            self.chat = DummyChat()

    service = chat_service.ChatService(system_prompt="You are helpful")
    service.client = DummyClient()

    response = asyncio.run(service.generate_response([Message(role="user", content="Who are you?")]))

    assert response.text == "I'm Alex. Nice to meet you. I enjoy talking with people and having interesting conversations."
    assert service.client.chat.completions.calls == []


def test_generate_response_returns_fallback_when_groq_fails(monkeypatch):
    class DummyCompletions:
        def create(self, model, messages, temperature, max_tokens):
            raise RuntimeError("Service unavailable")

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self):
            self.chat = DummyChat()

    service = chat_service.ChatService(system_prompt="You are helpful")
    service.client = DummyClient()

    response = asyncio.run(service.generate_response([Message(role="user", content="Hello")]))

    assert "unable to reach the ai service" in response.text.lower()


def test_validate_request_rejects_empty_messages():
    service = chat_service.ChatService(system_prompt="You are helpful")

    with pytest.raises(chat_service.ChatRequestError) as exc_info:
        service._validate_request([], 0.5)

    assert exc_info.value.status_code == 400
    assert "required" in str(exc_info.value).lower()


def test_initialize_rejects_missing_api_key(monkeypatch):
    monkeypatch.setattr(chat_service, "get_settings", lambda: type("Settings", (), {"GROQ_API_KEY": "", "GROQ_MODEL_NAME": "llama-3.3-70b-versatile"})())

    service = chat_service.ChatService(system_prompt="You are helpful")

    with pytest.raises(chat_service.ChatRequestError) as exc_info:
        service.initialize()

    assert exc_info.value.status_code == 500
    assert "api key" in str(exc_info.value).lower()
