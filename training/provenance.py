"""Dataset/code provenance helpers shared by training, Airflow and the agent.

Reads .git and .dvc files directly (no git/dvc binary needed), so it also works inside
the Airflow container where only the repository files are mounted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def parse_dvc_text(text: str) -> list[dict[str, Any]]:
    """Parse the content of a .dvc file into a list of tracked outputs."""
    doc = yaml.safe_load(text) or {}
    outs = doc.get("outs") or []
    return [
        {
            "path": o.get("path"),
            "md5": o.get("md5"),
            "size": o.get("size"),
            "nfiles": o.get("nfiles"),
        }
        for o in outs
        if isinstance(o, dict)
    ]


def read_dvc_file(path: Path) -> list[dict[str, Any]]:
    return parse_dvc_text(Path(path).read_text(encoding="utf-8"))


def read_git_commit(root: Path | str) -> str | None:
    """Return the current commit SHA by reading .git directly (fallback: $GIT_COMMIT)."""
    git_dir = Path(root) / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head  # detached HEAD
        ref = head.split(" ", 1)[1].strip()
        ref_file = git_dir / ref
        if ref_file.exists():
            return ref_file.read_text(encoding="utf-8").strip()
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
    except OSError:
        pass
    return os.getenv("GIT_COMMIT") or None


def dataset_provenance(root: Path | str, dvc_file: str = "data/train.csv.dvc") -> dict[str, Any]:
    """Provenance record stored as MLflow tags for every training run."""
    root = Path(root)
    md5 = None
    try:
        outs = read_dvc_file(root / dvc_file)
        md5 = outs[0]["md5"] if outs else None
    except (OSError, ValueError):
        pass
    return {"dvc_md5": md5, "git_commit": read_git_commit(root), "dvc_file": dvc_file}
