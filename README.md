# Turing Test Chatbot API

Production-ready FastAPI scaffold for a Turing Test chatbot.

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

## Project structure

```
app/
├── config/     # Settings and environment configuration
├── routes/     # HTTP endpoints
├── schemas/    # Pydantic request/response models
├── services/   # Business logic layer
└── utils/      # Shared utilities
```
