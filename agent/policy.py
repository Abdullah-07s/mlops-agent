"""Request-level policy: a state-changing tool may only run when the USER asked for it.

This is enforced by the orchestrator regardless of which planner (rules or LLM) proposed
the call, so a hallucinating or prompt-injected planner cannot start retraining or restart
a Deployment on its own.
"""

from __future__ import annotations

import re

_LEAD = r"(?:^|[.;,!?]|\band\b|\bthen\b)\s*(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?"
_TRIGGER = re.compile(
    _LEAD + r"(?:trigger|start|launch|kick\s*off|execute|run)\s+(?:the\s+|a\s+|an\s+)?[\w\s-]{0,30}?"
    r"(?:training|pipeline|dag|workflow)\b",
    re.I,
)
_RESTART = re.compile(
    _LEAD + r"(?:restart|bounce|roll(?:ing)?\s*restart)\s+(?:the\s+|my\s+)?[\w\s-]{0,30}?"
    r"(?:deployment|pods?|api|service)\b",
    re.I,
)

MUTATION_INTENTS = {
    "airflow_trigger_training_dag": _TRIGGER,
    "k8s_restart_deployment": _RESTART,
}


def explicit_trigger(message: str) -> bool:
    return bool(_TRIGGER.search(message))


def explicit_restart(message: str) -> bool:
    return bool(_RESTART.search(message))


def mutation_requested(tool: str, message: str) -> bool:
    """Unknown mutating tools are denied by default."""
    pattern = MUTATION_INTENTS.get(tool)
    return bool(pattern and pattern.search(message))
