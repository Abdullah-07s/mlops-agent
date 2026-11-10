"""Allow-listed subprocess execution. No shell, no free-form commands."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from agent.errors import ToolError

# (executable, subcommand) pairs. Read-only operations only.
ALLOWED_COMMANDS = frozenset(
    {
        ("git", "rev-parse"),
        ("git", "log"),
        ("git", "show"),
        ("git", "status"),
        ("dvc", "status"),
        ("dvc", "diff"),
    }
)

_REV_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./~^@{}-]{0,127}$")
_RELPATH_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]{0,200}$")

Runner = Callable[..., subprocess.CompletedProcess]


def validate_rev(rev: str) -> str:
    """Reject anything that could be parsed as an option or a revision range."""
    if not _REV_RE.match(rev) or ".." in rev:
        raise ToolError("invalid_revision", f"Invalid git revision: {rev!r}")
    return rev


def validate_relpath(path: str) -> str:
    candidate = path.replace("\\", "/")
    if not _RELPATH_RE.match(candidate) or any(p == ".." for p in candidate.split("/")):
        raise ToolError("invalid_path", f"Invalid repository-relative path: {path!r}")
    return candidate


def run_command(
    args: Sequence[str],
    *,
    cwd: Path | str,
    timeout: float = 20.0,
    runner: Runner | None = None,
) -> subprocess.CompletedProcess:
    if tuple(args[:2]) not in ALLOWED_COMMANDS:
        raise ToolError("command_not_allowed", f"Command not on the allow-list: {' '.join(args[:2])}")
    run = runner or subprocess.run
    try:
        return run(
            list(args),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ToolError("binary_not_found", f"'{args[0]}' is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError("command_timeout", f"'{args[0]} {args[1]}' timed out after {timeout}s") from exc
