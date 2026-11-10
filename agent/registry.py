"""Tool registry: validation, authorization, timeouts, error classification and logging.

Every tool call - whoever requested it (rules planner, Gemini, a test) - goes through
ToolRegistry.execute(), so safety controls cannot be bypassed by a planner.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from agent.errors import INVALID_CODES, REJECTION_CODES, ToolError
from agent.logging_utils import log_event, redact, redact_text
from agent.schemas import ToolCall, ToolResult, ToolStatus

logger = logging.getLogger("agent.registry")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[Any], dict[str, Any]]
    mutating: bool = False
    timeout_s: float | None = None


@dataclass(frozen=True)
class ExecutionContext:
    request_id: str
    confirmed: bool = False


class ToolRegistry:
    def __init__(self, default_timeout_s: float = 20.0, max_workers: int = 8) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._default_timeout_s = default_timeout_s
        self._pool = cf.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool")

    # ---- registration / discovery -------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._tools[spec.name] = spec

    def register_all(self, specs: list[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "mutating": s.mutating,
                "args_schema": s.args_model.model_json_schema().get("properties", {}),
            }
            for s in sorted(self._tools.values(), key=lambda s: s.name)
        ]

    # ---- execution ------------------------------------------------------------------
    def execute(self, call: ToolCall, ctx: ExecutionContext) -> ToolResult:
        started = time.perf_counter()

        def finish(
            status: ToolStatus,
            *,
            data: dict[str, Any] | None = None,
            error: str | None = None,
            code: str | None = None,
        ) -> ToolResult:
            result = ToolResult(
                tool=call.tool,
                status=status,
                args=redact(call.args),
                data=data or {},
                error=redact_text(error) if error else None,
                error_code=code,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            log_event(
                logger,
                "tool_call",
                logging.INFO if status == ToolStatus.OK else logging.WARNING,
                tool=result.tool,
                status=result.status.value,
                duration_ms=result.duration_ms,
                request_id=ctx.request_id,
                error_code=code,
                args=call.args,
            )
            return result

        spec = self._tools.get(call.tool)
        if spec is None:
            return finish(ToolStatus.INVALID, error=f"Unknown tool '{call.tool}'", code="unknown_tool")

        try:
            args = spec.args_model.model_validate(call.args)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors()
            )
            return finish(ToolStatus.INVALID, error=problems, code="invalid_arguments")

        if spec.mutating and not ctx.confirmed:
            return finish(
                ToolStatus.REJECTED,
                error=f"'{spec.name}' changes state and requires explicit user confirmation.",
                code="confirmation_required",
            )

        future = self._pool.submit(spec.func, args)
        try:
            data = future.result(timeout=spec.timeout_s or self._default_timeout_s)
        except cf.TimeoutError:
            future.cancel()
            limit = spec.timeout_s or self._default_timeout_s
            return finish(ToolStatus.TIMEOUT, error=f"Timed out after {limit}s", code="timeout")
        except ToolError as exc:
            if exc.code in REJECTION_CODES:
                status = ToolStatus.REJECTED
            elif exc.code in INVALID_CODES:
                status = ToolStatus.INVALID
            else:
                status = ToolStatus.ERROR
            return finish(status, error=exc.message, code=exc.code)
        except Exception as exc:  # noqa: BLE001 - last line of defence for third-party errors
            logger.debug("unexpected tool failure", exc_info=True)
            return finish(
                ToolStatus.ERROR, error=f"{type(exc).__name__}: {exc}"[:500], code="internal_error"
            )

        return finish(ToolStatus.OK, data=data if isinstance(data, dict) else {"result": data})
