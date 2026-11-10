"""Pydantic models that cross module boundaries (tool I/O and the HTTP contract)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    INVALID = "invalid_request"


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool: str
    status: ToolStatus
    args: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    error_code: str | None = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.OK


class Fact(BaseModel):
    source: str
    statement: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    confirm_mutations: bool = False
    degradation_note: str | None = Field(default=None, max_length=1000)


class AgentResponse(BaseModel):
    request_id: str
    answer: str
    facts: list[Fact] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    tool_trace: list[ToolResult] = Field(default_factory=list)
