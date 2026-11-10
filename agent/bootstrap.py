"""Composition root: the only place that wires concrete clients together."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.config import Settings
from agent.llm import GeminiLLM, LLMClient, RuleBasedLLM
from agent.orchestrator import Agent
from agent.registry import ToolRegistry
from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools


def build_registry(
    settings: Settings,
    *,
    mlflow_client_factory: Callable[[], Any] | None = None,
    k8s_clients_factory: Callable[[], Any] | None = None,
    airflow_client: Any | None = None,
    dvc_runner: Any | None = None,
    sleep: Callable[[float], None] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(default_timeout_s=settings.tool_timeout_s)
    registry.register_all(dvc_tools.build_tools(settings, runner=dvc_runner))
    registry.register_all(mlflow_tools.build_tools(settings, client_factory=mlflow_client_factory))
    air_kwargs = {"sleep": sleep} if sleep else {}
    registry.register_all(airflow_tools.build_tools(settings, client=airflow_client, **air_kwargs))
    registry.register_all(kubernetes_tools.build_tools(settings, clients_factory=k8s_clients_factory))
    return registry


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "gemini":
        return GeminiLLM(settings.gemini_api_key, settings.gemini_model)
    return RuleBasedLLM(settings)


def build_agent(settings: Settings | None = None, **registry_kwargs: Any) -> Agent:
    settings = settings or Settings.from_env()
    return Agent(build_registry(settings, **registry_kwargs), settings, build_llm(settings))
