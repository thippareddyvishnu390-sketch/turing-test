import json
import logging

from app.utils.logging import StructuredFormatter


def test_structured_formatter_redacts_sensitive_fields():
    record = logging.LogRecord(
        name="app.routes.chat",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request received",
        args=(),
        exc_info=None,
    )
    record.event = "request_received"
    record.request_id = "req-123"
    record.api_key = "super-secret"

    payload = json.loads(StructuredFormatter().format(record))

    assert payload["event"] == "request_received"
    assert payload["request_id"] == "req-123"
    assert payload["api_key"] == "[REDACTED]"
    assert payload["message"] == "request received"
