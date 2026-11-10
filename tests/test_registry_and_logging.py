from __future__ import annotations

import logging
import time

from pydantic import BaseModel

from agent.logging_utils import redact
from agent.registry import ExecutionContext, ToolRegistry, ToolSpec
from agent.schemas import ToolCall, ToolStatus


class Args(BaseModel):
    x: int = 1


def _reg(func, **kw):
    reg = ToolRegistry(default_timeout_s=0.2)
    reg.register(ToolSpec("t", "test tool", Args, func, **kw))
    return reg


CTX = ExecutionContext("req-1", confirmed=False)


def test_unknown_tool_is_invalid():
    r = _reg(lambda a: {}).execute(ToolCall(tool="nope"), CTX)
    assert r.status == ToolStatus.INVALID and r.error_code == "unknown_tool"


def test_invalid_arguments_rejected_before_execution():
    called = []
    r = _reg(lambda a: called.append(1) or {}).execute(ToolCall(tool="t", args={"x": "abc"}), CTX)
    assert r.status == ToolStatus.INVALID and not called


def test_tool_timeout_is_reported_not_raised():
    r = _reg(lambda a: time.sleep(1) or {}).execute(ToolCall(tool="t"), CTX)
    assert r.status == ToolStatus.TIMEOUT and r.error_code == "timeout"


def test_service_failure_is_classified_and_secret_not_leaked():
    def boom(a):
        raise RuntimeError("connect failed password=hunter2")

    r = _reg(boom).execute(ToolCall(tool="t"), CTX)
    assert r.status == ToolStatus.ERROR
    assert "hunter2" not in (r.error or "")


def test_mutating_tool_needs_confirmation():
    reg = _reg(lambda a: {"done": True}, mutating=True)
    assert reg.execute(ToolCall(tool="t"), CTX).error_code == "confirmation_required"
    assert reg.execute(ToolCall(tool="t"), ExecutionContext("r", confirmed=True)).ok


def test_structured_log_has_required_fields_and_no_secrets(caplog):
    caplog.set_level(logging.INFO, logger="agent")
    _reg(lambda a: {}).execute(ToolCall(tool="t", args={"x": 2}), CTX)
    rec = next(r for r in caplog.records if r.getMessage() == "tool_call")
    assert rec.fields["tool"] == "t" and rec.fields["status"] == "ok"
    assert rec.fields["request_id"] == "req-1" and "duration_ms" in rec.fields


def test_redact_masks_keys_and_values():
    out = redact({"password": "x", "n": {"api_key": "y"}, "msg": "Bearer abc.def.ghi"})
    assert out["password"] == out["n"]["api_key"] == "***REDACTED***"
    assert "abc.def.ghi" not in out["msg"]
