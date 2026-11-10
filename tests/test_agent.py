from __future__ import annotations

import requests

from agent.errors import ToolError
from agent.llm import Action, GeminiLLM, RuleBasedLLM
from agent.schemas import ChatRequest, ToolStatus
from tests import fakes

SCENARIO = ("Find the best recent model, identify the dataset version used to train it, "
            "check the deployment status, and explain whether retraining is necessary.")


def tools(resp):
    return [r.tool for r in resp.tool_trace]


def test_mandatory_multi_tool_scenario(make_agent):
    agent, h = make_agent()
    resp = agent.handle(ChatRequest(message=SCENARIO))
    called = tools(resp)
    for expected in ("mlflow_best_run", "dvc_find_commit_for_hash", "k8s_get_deployed_version",
                     "k8s_find_unhealthy_pods", "airflow_get_dag_runs"):
        assert expected in called
    assert "airflow_trigger_training_dag" not in called
    assert not h["session"].posted(f"{fakes.DAG}/dagRuns")
    text = " ".join(f.statement for f in resp.facts)
    assert "run-best" in text and fakes.COMMIT in text
    assert any("No retraining trigger" in r for r in resp.recommendations)
    assert all(r.ok for r in resp.tool_trace)


def test_dataset_provenance_for_experiment(make_agent):
    agent, _ = make_agent()
    resp = agent.handle(ChatRequest(message="Which dataset version was used for experiment 1?"))
    assert tools(resp)[:2] == ["mlflow_list_runs", "dvc_current_version"]
    assert "dvc_find_commit_for_hash" in tools(resp)


def test_model_comparison_request(make_agent):
    agent, _ = make_agent()
    resp = agent.handle(ChatRequest(message="Compare the last 2 training runs and recommend the best candidate based on F1 score"))
    assert "mlflow_compare_runs" in tools(resp)


def test_failed_quality_gate_recommends_retraining_but_never_starts_it(make_agent):
    session = fakes.FakeSession(fakes.airflow_routes(latest_state="failed", run_id="manual__2", failed_task="apply_quality_gate"))
    agent, h = make_agent(session=session)
    resp = agent.handle(ChatRequest(message="Is retraining necessary?"))
    assert "airflow_get_task_logs" in tools(resp)
    assert any("Retraining may be warranted (NOT started automatically)" in r and "apply_quality_gate" in r
               for r in resp.recommendations)
    assert not h["session"].posted(f"{fakes.DAG}/dagRuns")


def test_user_supplied_degradation_justifies_recommendation(make_agent):
    agent, _ = make_agent()
    resp = agent.handle(ChatRequest(message="Is retraining necessary?", degradation_note="precision fell 12% this week"))
    assert any("user-supplied degradation" in r for r in resp.recommendations)


def test_low_metric_below_policy_recommends_retraining(make_agent):
    agent, _ = make_agent(runs=[fakes.make_run("weak-run", 0.60)])
    resp = agent.handle(ChatRequest(message="Is retraining necessary?"))
    assert any("below the policy threshold" in r for r in resp.recommendations)


def test_agent_does_not_invent_results_when_tools_are_unavailable(settings, tmp_path, make_agent):
    import dataclasses

    from agent.tools.airflow_tools import AirflowClient

    bare = dataclasses.replace(settings, repo_root=str(tmp_path / "empty"))  # no .git => DVC tools fail
    (tmp_path / "empty").mkdir()

    class BrokenMlflow:
        def __getattr__(self, name):
            raise requests.exceptions.ConnectionError("mlflow down")

    def no_cluster():
        raise ToolError("k8s_not_configured", "no kubeconfig")

    from agent.bootstrap import build_agent

    def missing_binary(*a, **kw):
        raise FileNotFoundError("dvc")

    agent = build_agent(
        bare,
        mlflow_client_factory=lambda: BrokenMlflow(),
        k8s_clients_factory=no_cluster,
        airflow_client=AirflowClient(bare.airflow_base_url, "airflow", "pw", session=fakes.DownSession()),
        dvc_runner=missing_binary,
    )
    resp = agent.handle(ChatRequest(message=SCENARIO))
    assert resp.facts == [] and resp.recommendations == []
    assert len(resp.uncertainties) >= 4
    assert all(r.status != ToolStatus.OK for r in resp.tool_trace)
    assert "None: no tool returned verifiable data" in resp.answer
    assert "0.9" not in resp.answer  # no metric values can appear without evidence


def test_diagnosis_of_unavailable_api(make_agent):
    pods = [fakes.make_pod("p", phase="Pending", ready=False, waiting="ImagePullBackOff")]
    agent, _ = make_agent(k8s=fakes.k8s_factory(pods=pods, deployment=fakes.make_deployment(ready=0)))
    resp = agent.handle(ChatRequest(message="Why is my prediction API unavailable?"))
    assert "k8s_diagnose_deployment" in tools(resp)
    assert any("image cannot be pulled" in f.statement for f in resp.facts)


def test_trigger_requires_explicit_request_and_confirmation(make_agent):
    agent, h = make_agent()
    blocked = agent.handle(ChatRequest(message="Trigger the model training pipeline and report whether it succeeded"))
    trig = next(r for r in blocked.tool_trace if r.tool == "airflow_trigger_training_dag")
    assert trig.error_code == "confirmation_required" and not h["session"].posted(f"{fakes.DAG}/dagRuns")

    done = agent.handle(ChatRequest(message="Trigger the model training pipeline and report whether it succeeded",
                                    confirm_mutations=True))
    trig = next(r for r in done.tool_trace if r.tool == "airflow_trigger_training_dag")
    assert trig.ok and trig.data["state"] == "success"
    assert "mlflow_best_run" not in tools(done)


class RogueLLM(RuleBasedLLM):
    """A planner that tries to retrain and restart without being asked."""
    name = "rogue"

    def __init__(self, settings):
        super().__init__(settings)
        self._queue = ["airflow_trigger_training_dag", "k8s_restart_deployment"]

    def next_action(self, question, history, tools, request):
        if self._queue:
            return Action(kind="tool", tool=self._queue.pop(0), args={})
        return Action(kind="final")


def test_planner_cannot_trigger_mutations_the_user_did_not_request(settings, make_agent):
    agent, h = make_agent(llm=RogueLLM(settings))
    resp = agent.handle(ChatRequest(message="How is the system doing?", confirm_mutations=True))
    assert [r.error_code for r in resp.tool_trace] == ["not_requested", "not_requested"]
    assert h["k8s"].apps.patches == [] and not h["session"].posted(f"{fakes.DAG}/dagRuns")


class InventingLLM(RuleBasedLLM):
    name = "inventing"

    def write_answer(self, question, facts, uncertainties, recommendations):
        return "The model reached F1 0.9999 on 42 pods."


def test_ungrounded_llm_narrative_is_replaced_by_evidence_report(settings, make_agent):
    agent, _ = make_agent(llm=InventingLLM(settings))
    resp = agent.handle(ChatRequest(message="check the deployment"))
    assert "0.9999" not in resp.answer and resp.answer.startswith("Known facts")


def test_gemini_outage_falls_back_to_rules(settings, make_agent):
    class Session:
        def post(self, *a, **kw):
            raise requests.exceptions.ConnectionError("offline")

    agent, _ = make_agent(llm=GeminiLLM("key", "model", session=Session()))
    resp = agent.handle(ChatRequest(message="check the deployment"))
    assert "k8s_get_deployed_version" in tools(resp) and resp.answer.startswith("Known facts")


def test_gemini_planner_json_is_parsed(settings):
    class Session:
        def post(self, *a, **kw):
            body = {"candidates": [{"content": {"parts": [{"text": '```json\n{"action":"tool","tool":"dvc_list_datasets","args":{}}\n```'}]}}]}
            return type("R", (), {"status_code": 200, "json": lambda self: body, "text": ""})()

    action = GeminiLLM("k", "m", session=Session()).next_action("q", [], [], ChatRequest(message="q"))
    assert action.tool == "dvc_list_datasets"


def test_unrelated_question_calls_no_tools(make_agent):
    agent, _ = make_agent()
    resp = agent.handle(ChatRequest(message="hello there"))
    assert resp.tool_trace == [] and resp.facts == []


def test_agent_never_loops_forever(settings, make_agent):
    class Looper(RuleBasedLLM):
        name = "looper"

        def next_action(self, *a, **k):
            return Action(kind="tool", tool="dvc_list_datasets", args={})

    agent, _ = make_agent(llm=Looper(settings))
    assert len(agent.handle(ChatRequest(message="x")).tool_trace) == 1
