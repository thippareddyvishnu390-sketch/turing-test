# Turing Test Chatbot API

Production-ready FastAPI scaffold for a Turing Test chatbot.

## System prompt handling

- The system prompt is loaded once at startup and cached in memory.
- Each request reuses the cached prompt rather than reloading the file.
- Groq receives the prompt through the OpenAI-style `messages` payload; the cached system prompt is inserted as the first message for every request.
- The /chat/completions endpoint remains OpenAI-compatible.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Run

```bash
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

## Production deployment

- Copy [.env.example](.env.example) to .env and set your Gemini credentials.
- For container deployment, build and run the included Docker image:

```bash
docker build -t turing-test-api .
docker run -p 8000:8000 --env-file .env turing-test-api
```

- The app preserves the OpenAI-compatible /chat/completions contract while using runtime configuration for environment, CORS, and health metadata.

## Project structure

```
app/
├── config/     # Settings and environment configuration
├── routes/     # HTTP endpoints
├── schemas/    # Pydantic request/response models
├── services/   # Business logic layer
└── utils/      # Shared utilities
```
