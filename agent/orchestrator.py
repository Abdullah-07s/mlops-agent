"""The agent loop: plan -> (policy) -> execute -> repeat -> correlate -> answer."""

from __future__ import annotations

import json
import logging
import time

from agent import correlator, policy
from agent.config import Settings
from agent.llm import LLMClient, LLMUnavailable, RuleBasedLLM
from agent.logging_utils import log_event, new_request_id, request_id_var
from agent.registry import ExecutionContext, ToolRegistry
from agent.report import is_grounded, render_report
from agent.schemas import AgentResponse, ChatRequest, ToolCall, ToolResult, ToolStatus

logger = logging.getLogger("agent.orchestrator")


class Agent:
    def __init__(
        self,
        registry: ToolRegistry,
        settings: Settings,
        llm: LLMClient,
        fallback: LLMClient | None = None,
    ) -> None:
        self.registry = registry
        self.settings = settings
        self.llm = llm
        self.fallback = fallback or RuleBasedLLM(settings)

    def handle(self, request: ChatRequest, request_id: str | None = None) -> AgentResponse:
        rid = request_id or new_request_id()
        token = request_id_var.set(rid)
        try:
            return self._run(request, rid)
        finally:
            request_id_var.reset(token)

    # ------------------------------------------------------------------------------
    def _run(self, request: ChatRequest, rid: str) -> AgentResponse:
        started = time.perf_counter()
        ctx = ExecutionContext(request_id=rid, confirmed=request.confirm_mutations)
        catalog = self.registry.catalog()
        history: list[ToolResult] = []
        seen: set[str] = set()

        for _ in range(self.settings.max_agent_steps):
            action = self._decide(request, history, catalog)
            if action.kind == "final" or not action.tool:
                break
            call = ToolCall(tool=action.tool, args=action.args)
            key = json.dumps(call.model_dump(), sort_keys=True, default=str)
            if key in seen:  # loop protection: never repeat an identical call
                break
            seen.add(key)
            history.append(self._execute(call, request, ctx))

        facts = correlator.facts_from(history)
        recs, extra_unc = correlator.recommend(history, request, self.settings)
        uncertainties = correlator.uncertainties_from(history) + extra_unc
        answer = self._write(request.message, facts, uncertainties, recs)

        log_event(
            logger,
            "agent_request_completed",
            request_id=rid,
            tools_called=[r.tool for r in history],
            failures=sum(1 for r in history if not r.ok),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return AgentResponse(
            request_id=rid,
            answer=answer,
            facts=facts,
            uncertainties=uncertainties,
            recommendations=recs,
            tool_trace=history,
        )

    def _decide(self, request, history, catalog):
        try:
            return self.llm.next_action(request.message, history, catalog, request)
        except LLMUnavailable as exc:
            log_event(logger, "llm_fallback", logging.WARNING, reason=str(exc), stage="plan")
            return self.fallback.next_action(request.message, history, catalog, request)

    def _execute(self, call: ToolCall, request: ChatRequest, ctx: ExecutionContext) -> ToolResult:
        spec = self.registry.get(call.tool)
        if spec and spec.mutating and not policy.mutation_requested(call.tool, request.message):
            result = ToolResult(
                tool=call.tool,
                status=ToolStatus.REJECTED,
                args=call.args,
                error="The user did not explicitly ask for this state-changing action.",
                error_code="not_requested",
            )
            log_event(logger, "tool_call", logging.WARNING, tool=call.tool, status="rejected",
                      error_code="not_requested", duration_ms=0.0, request_id=ctx.request_id)
            return result
        return self.registry.execute(call, ctx)

    def _write(self, question, facts, uncertainties, recs) -> str:
        grounding = [f.statement for f in facts] + uncertainties + recs + [question]
        try:
            text = self.llm.write_answer(question, facts, uncertainties, recs)
            if self.llm.name != "rules" and not is_grounded(text, grounding):
                raise LLMUnavailable("narrative contained numbers that are not in the evidence")
            return text
        except LLMUnavailable as exc:
            log_event(logger, "llm_fallback", logging.WARNING, reason=str(exc), stage="write")
            return render_report(facts, uncertainties, recs)
