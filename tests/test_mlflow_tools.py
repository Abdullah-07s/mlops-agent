from __future__ import annotations

from agent.registry import ExecutionContext, ToolRegistry
from agent.schemas import ToolCall, ToolStatus
from agent.tools import mlflow_tools
from tests import fakes
from tests.fakes import make_run

CTX = ExecutionContext("t")


def call(settings, runs, tool, **args):
    r = ToolRegistry()
    r.register_all(mlflow_tools.build_tools(settings, client_factory=lambda: fakes.FakeMlflowClient(runs)))
    return r.execute(ToolCall(tool=tool, args=args), CTX)


RUNS = [make_run("a", 0.80, start=1), make_run("b", 0.95, start=2, gate=None), make_run("c", 0.90, start=3),
        make_run("d", 0.99, start=4, status="FAILED")]


def test_best_run_ignores_failed_runs_and_flags_missing_gate(settings):
    res = call(settings, RUNS, "mlflow_best_run")
    assert res.data["best"]["run_id"] == "b" and res.data["considered"] == 3
    assert any("quality-gate" in c for c in res.data["caveats"])


def test_best_run_min_mode(settings):
    assert call(settings, RUNS, "mlflow_best_run", mode="min").data["best"]["run_id"] == "a"


def test_unknown_metric_is_a_clear_error(settings):
    res = call(settings, RUNS, "mlflow_best_run", metric="auc")
    assert res.status == ToolStatus.ERROR and res.error_code == "no_runs_with_metric"


def test_compare_last_runs(settings):
    res = call(settings, RUNS, "mlflow_compare_runs", last_n=3)
    assert res.data["winner"]["run_id"] == "b" and len(res.data["table"]) == 3


def test_list_runs_by_numeric_experiment_id(settings):
    assert call(settings, RUNS, "mlflow_list_runs", experiment="1", limit=2).ok
    assert call(settings, RUNS, "mlflow_list_runs", experiment="25").error_code == "experiment_not_found"


def test_get_run_returns_provenance(settings):
    assert call(settings, RUNS, "mlflow_get_run", run_id="c").status == ToolStatus.INVALID  # id too short
    res = call(settings, [make_run("run-123456", 0.9)], "mlflow_get_run", run_id="run-123456")
    assert res.data["run"]["provenance"]["dvc_md5"] == fakes.MD5


def test_registry_info(settings):
    from types import SimpleNamespace as NS
    r = ToolRegistry()
    client = fakes.FakeMlflowClient(RUNS, models=[NS(name="m", aliases={"candidate": "1"})])
    r.register_all(mlflow_tools.build_tools(settings, client_factory=lambda: client))
    res = r.execute(ToolCall(tool="mlflow_get_registry_info"), CTX)
    assert res.data["models"][0]["versions"][0]["aliases"] == ["candidate"]


def test_ties_are_reported_as_caveat(settings):
    tied = [make_run("x1", 0.97, start=1), make_run("x2", 0.97, start=2), make_run("x3", 0.90, start=3)]
    best = call(settings, tied, "mlflow_best_run")
    assert best.data["best"]["run_id"] == "x2"  # newest of the tied runs
    assert any("2 runs tie" in c for c in best.data["caveats"])
    cmp_ = call(settings, tied, "mlflow_compare_runs", last_n=3)
    assert any("tie" in c for c in cmp_.data["caveats"])