from __future__ import annotations

from agent.registry import ExecutionContext, ToolRegistry
from agent.schemas import ToolCall, ToolStatus
from agent.tools import airflow_tools
from agent.tools.airflow_tools import AirflowClient
from tests import fakes

OK = ExecutionContext("t", confirmed=True)


def make(settings, session):
    r = ToolRegistry()
    client = AirflowClient(settings.airflow_base_url, "airflow", "pw", session=session)
    r.register_all(airflow_tools.build_tools(settings, client=client, sleep=lambda s: None))
    return r


def test_authenticates_then_lists_runs(settings):
    s = fakes.FakeSession(fakes.airflow_routes())
    res = make(settings, s).execute(ToolCall(tool="airflow_get_dag_runs"), OK)
    assert res.data["latest_state"] == "success"
    assert s.calls[0][1] == "/auth/token"
    assert s.calls[1][2]["headers"]["Authorization"] == "Bearer jwt.token.value"


def test_trigger_unapproved_dag_rejected(settings):
    s = fakes.FakeSession(fakes.airflow_routes())
    res = make(settings, s).execute(ToolCall(tool="airflow_trigger_training_dag", args={"dag_id": "other_dag"}), OK)
    assert res.status == ToolStatus.REJECTED and res.error_code == "dag_not_approved"
    assert not s.posted("/api/v2/dags/other_dag/dagRuns")


def test_trigger_requires_confirmation(settings):
    s = fakes.FakeSession(fakes.airflow_routes())
    res = make(settings, s).execute(ToolCall(tool="airflow_trigger_training_dag"), ExecutionContext("t"))
    assert res.error_code == "confirmation_required" and not s.posted(f"{fakes.DAG}/dagRuns")


def test_trigger_posts_and_waits_for_terminal_state(settings):
    s = fakes.FakeSession(fakes.airflow_routes())
    res = make(settings, s).execute(ToolCall(tool="airflow_trigger_training_dag", args={"wait_seconds": 10}), OK)
    assert res.data["state"] == "success" and res.data["finished"]
    assert s.posted(f"{fakes.DAG}/dagRuns")[0]["json"] == {"logical_date": None, "conf": {}}


def test_failed_tasks_and_logs(settings):
    s = fakes.FakeSession(fakes.airflow_routes(latest_state="failed", run_id="manual__2", failed_task="apply_quality_gate"))
    r = make(settings, s)
    failed = r.execute(ToolCall(tool="airflow_get_failed_tasks"), OK)
    assert [t["task_id"] for t in failed.data["failed"]] == ["apply_quality_gate"]
    logs = r.execute(ToolCall(tool="airflow_get_task_logs", args={
        "dag_id": "training_pipeline", "dag_run_id": "manual__2", "task_id": "apply_quality_gate"}), OK)
    assert "below threshold" in logs.data["log_tail"]


def test_unreachable_airflow_is_a_tool_error(settings):
    res = make(settings, fakes.DownSession()).execute(ToolCall(tool="airflow_list_dags"), OK)
    assert res.status == ToolStatus.ERROR and res.error_code == "airflow_unreachable"


def test_missing_password_is_reported(settings):
    r = ToolRegistry()
    client = AirflowClient("http://airflow.test", "airflow", "", session=fakes.FakeSession())
    r.register_all(airflow_tools.build_tools(settings, client=client))
    assert r.execute(ToolCall(tool="airflow_list_dags"), OK).error_code == "airflow_not_configured"
