"""Structured JSON logging with request correlation and secret redaction."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

REDACTED = "***REDACTED***"
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_SENSITIVE_KEY = re.compile(r"(pass(word)?|secret|token|api[_-]?key|authorization|credential)", re.I)
_SENSITIVE_VALUE = re.compile(
    r"(Bearer\s+\S+"
    r"|AIza[0-9A-Za-z_\-]{20,}"
    r"|eyJ[\w\-]{8,}\.[\w\-]{8,}\.[\w\-]{8,}"
    r"|(?i:password|passwd|token|secret|api[_-]?key)\s*[=:]\s*\S+)"
)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def redact_text(text: str) -> str:
    return _SENSITIVE_VALUE.sub(REDACTED, text)


def redact(obj: Any) -> Any:
    """Recursively mask sensitive keys and secret-looking values."""
    if isinstance(obj, dict):
        return {
            k: (REDACTED if _SENSITIVE_KEY.search(str(k)) else redact(v)) for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": redact(fields)})
