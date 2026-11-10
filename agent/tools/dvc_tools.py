"""Task 1 - DVC data management tools (read-only, allow-listed commands)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.config import Settings
from agent.errors import ToolError
from agent.logging_utils import redact_text
from agent.registry import ToolSpec
from agent.tools.safe_exec import Runner, run_command, validate_relpath, validate_rev
from training.provenance import parse_dvc_text, read_dvc_file, read_git_commit

_SKIP_DIRS = {".git", ".dvc", ".venv", "venv", "node_modules", "__pycache__", "mlartifacts"}


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DvcFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dvc_file: str | None = None


class CompareArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rev_a: str
    rev_b: str
    dvc_file: str | None = None


class FindCommitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    md5: str = Field(pattern=r"^[a-f0-9]{32}(\.dir)?$")
    dvc_file: str | None = None


class ReproduceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rev: str = "HEAD"
    dvc_file: str | None = None


def _iter_dvc_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in sorted(filenames):
            if name.endswith(".dvc"):
                yield Path(dirpath) / name


def build_tools(settings: Settings, runner: Runner | None = None) -> list[ToolSpec]:
    root = Path(settings.repo_root)

    def git(*args: str):
        return run_command(["git", *args], cwd=root, runner=runner)

    def dvc_rel(value: str | None) -> str:
        rel = validate_relpath(value or settings.dvc_file)
        if not rel.endswith(".dvc"):
            raise ToolError("invalid_path", f"Not a .dvc file: {rel}")
        return rel

    def datasets() -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in _iter_dvc_files(root):
            rel = path.relative_to(root).as_posix()
            for out in read_dvc_file(path):
                md5 = out.get("md5") or ""
                items.append({"dvc_file": rel, **out, "version_id": md5[:8]})
        return items

    def at_revision(rev: str, rel: str) -> dict[str, Any]:
        rev = validate_rev(rev)
        shown = git("show", f"{rev}:{rel}")
        if shown.returncode != 0:
            raise ToolError(
                "not_found_at_revision",
                f"{rel} not found at {rev}: {redact_text(shown.stderr.strip()[:200])}",
            )
        outs = parse_dvc_text(shown.stdout)
        if not outs:
            raise ToolError("no_outputs", f"{rel} at {rev} declares no tracked outputs")
        resolved = git("rev-parse", "--verify", f"{rev}^{{commit}}")
        commit = resolved.stdout.strip() if resolved.returncode == 0 else rev
        return {"rev": rev, "commit": commit, **outs[0]}

    def list_datasets(_: NoArgs) -> dict[str, Any]:
        items = datasets()
        return {"datasets": items, "count": len(items)}

    def current_version(_: NoArgs) -> dict[str, Any]:
        commit = read_git_commit(root)
        if not commit:
            raise ToolError("not_a_git_repo", f"No git commit found under {root.resolve()}")
        dirty: list[str] | None = None
        try:
            status = git("status", "--porcelain")
            if status.returncode == 0:
                dirty = [ln[3:] for ln in status.stdout.splitlines() if ln.strip()]
        except ToolError:
            dirty = None  # git binary unavailable: report "unknown", never guess
        return {
            "git_commit": commit,
            "git_dirty": None if dirty is None else bool(dirty),
            "dirty_paths": (dirty or [])[:20],
            "datasets": datasets(),
        }

    def compare_versions(a: CompareArgs) -> dict[str, Any]:
        rel = dvc_rel(a.dvc_file)
        left, right = at_revision(a.rev_a, rel), at_revision(a.rev_b, rel)
        sizes = (left.get("size"), right.get("size"))
        return {
            "dvc_file": rel,
            "a": left,
            "b": right,
            "changed": left.get("md5") != right.get("md5"),
            "size_delta_bytes": (sizes[1] - sizes[0]) if all(isinstance(s, int) for s in sizes) else None,
        }

    def detect_changes(_: NoArgs) -> dict[str, Any]:
        proc = run_command(["dvc", "status", "--json"], cwd=root, runner=runner)
        if proc.returncode != 0:
            raise ToolError("dvc_status_failed", redact_text(proc.stderr.strip()[:300]) or "dvc status failed")
        text = proc.stdout.strip()
        try:
            details: Any = json.loads(text) if text else {}
        except json.JSONDecodeError:
            details = {"raw": text[:500]}
        if isinstance(details, dict) and "raw" in details:
            changed = "up to date" not in details["raw"].lower()
        else:
            changed = bool(details)
        return {"changed": changed, "details": details}

    def find_commit(a: FindCommitArgs) -> dict[str, Any]:
        rel = dvc_rel(a.dvc_file)
        proc = git("log", "--format=%H%x09%cI%x09%s", f"-S{a.md5}", "--", rel)
        if proc.returncode != 0:
            raise ToolError("git_log_failed", redact_text(proc.stderr.strip()[:300]))
        commits = []
        for line in proc.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                commits.append({"commit": parts[0], "committed_at": parts[1], "subject": parts[2]})
        return {
            "md5": a.md5,
            "dvc_file": rel,
            "found": bool(commits),
            "commits": commits,
            # git log is newest-first, so the oldest entry is the commit that introduced the hash
            "introducing_commit": commits[-1]["commit"] if commits else None,
        }

    def reproduce_steps(a: ReproduceArgs) -> dict[str, Any]:
        rel = dvc_rel(a.dvc_file)
        info = at_revision(a.rev, rel)
        return {
            "target": info,
            "steps": [
                f"git switch --detach {info['commit']}",
                f"dvc pull {rel}   # use 'dvc checkout {rel}' if the data is already in your local cache",
                "dvc status       # expected: Data and pipelines are up to date.",
                "git switch -     # return to your previous branch",
                "dvc pull         # restore the data that matches the branch again",
            ],
            "notes": [
                "These commands are returned for a human to run; the agent never changes the workspace.",
                f"Expected content hash in {rel}: {info.get('md5')}",
            ],
        }

    return [
        ToolSpec("dvc_list_datasets", "List datasets tracked by DVC (.dvc files) with content hashes.", NoArgs, list_datasets),
        ToolSpec("dvc_current_version", "Current git commit, working-tree state and DVC dataset hashes.", NoArgs, current_version),
        ToolSpec("dvc_compare_versions", "Compare a dataset between two git revisions (hash and size).", CompareArgs, compare_versions),
        ToolSpec("dvc_detect_changes", "Detect whether DVC-tracked data or pipeline stages changed (dvc status).", NoArgs, detect_changes),
        ToolSpec("dvc_find_commit_for_hash", "Find the git commit that introduced a given DVC dataset hash.", FindCommitArgs, find_commit),
        ToolSpec("dvc_reproduce_steps", "Explain how to restore the dataset exactly as it was at a git revision.", ReproduceArgs, reproduce_steps),
    ]
