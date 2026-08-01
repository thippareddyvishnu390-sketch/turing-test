import json
import logging
import sys
import time
from typing import Any, Final

# Constants for logging configuration
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
REDACTED: Final[str] = "[REDACTED]"
SENSITIVE_FIELDS: Final[set[str]] = {"api_key", "apikey", "token", "authorization", "secret", "password"}


class StructuredFormatter(logging.Formatter):
    """Serialize log records as JSON while redacting sensitive fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in {"name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process", "message", "asctime"}:
                continue
            if key in {"api_key", "apikey", "token", "authorization", "secret", "password"}:
                payload[key] = REDACTED
            else:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, sort_keys=True)


def setup_logging() -> None:
    """Configure the application logger to emit structured JSON logs to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [handler]
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Retrieve a logger instance for the specified module."""
    return logging.getLogger(name)


def log_request_started(logger: logging.Logger, request_id: str, method: str, path: str, model: str | None = None, **context: Any) -> None:
    logger.info(
        "request received",
        extra={
            "event": "request_received",
            "request_id": request_id,
            "method": method,
            "path": path,
            "model": model,
            **context,
        },
    )


def log_request_completed(logger: logging.Logger, request_id: str, duration_ms: float, model: str | None = None, **context: Any) -> None:
    logger.info(
        "request completed",
        extra={
            "event": "request_completed",
            "request_id": request_id,
            "duration_ms": round(duration_ms, 3),
            "model": model,
            **context,
        },
    )


def log_api_failure(logger: logging.Logger, request_id: str, message: str, model: str | None = None, **context: Any) -> None:
    logger.exception(
        message,
        extra={
            "event": "api_failure",
            "request_id": request_id,
            "model": model,
            **context,
        },
    )


def log_startup(logger: logging.Logger, **context: Any) -> None:
    logger.info(
        "application startup",
        extra={"event": "startup", **context},
    )


def log_shutdown(logger: logging.Logger, **context: Any) -> None:
    logger.info(
        "application shutdown",
        extra={"event": "shutdown", **context},
    )


def time_request(logger: logging.Logger, request_id: str, method: str, path: str, model: str | None = None, **context: Any) -> Any:
    start = time.perf_counter()
    log_request_started(logger, request_id, method, path, model=model, **context)
    return start


# Initialize logging configuration on module load
setup_logging()