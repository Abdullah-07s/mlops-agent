"""Rendering and grounding helpers for the final answer."""

from __future__ import annotations

import re
from collections.abc import Iterable

from agent.schemas import Fact

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+", re.M)


def render_report(facts: list[Fact], uncertainties: list[str], recommendations: list[str]) -> str:
    lines = ["Known facts (from tool responses):"]
    lines += [f"- [{f.source}] {f.statement}" for f in facts] or [
        "- None: no tool returned verifiable data."
    ]
    lines += ["", "Uncertainties:"] + ([f"- {u}" for u in uncertainties] or ["- None identified."])
    lines += ["", "Recommended actions:"] + ([f"- {r}" for r in recommendations] or ["- None."])
    return "\n".join(lines)


def is_grounded(text: str, allowed_sources: Iterable[str]) -> bool:
    """Every number in an LLM-written narrative must already appear in the evidence."""
    allowed = set(_NUMBER.findall(" ".join(allowed_sources)))
    used = set(_NUMBER.findall(_LIST_MARKER.sub("", text)))
    return used <= allowed
