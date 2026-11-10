"""Typed errors shared by the tool modules and the registry."""

from __future__ import annotations

# Codes that mean "the safety layer said no" (reported as status=rejected).
REJECTION_CODES = frozenset(
    {
        "confirmation_required",
        "namespace_not_allowed",
        "dag_not_approved",
        "command_not_allowed",
        "not_requested",
    }
)

# Codes that mean "the request itself was malformed" (status=invalid_request).
INVALID_CODES = frozenset({"invalid_revision", "invalid_path", "invalid_arguments", "unknown_tool"})


class ToolError(Exception):
    """Expected, classified failure raised by a tool implementation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
