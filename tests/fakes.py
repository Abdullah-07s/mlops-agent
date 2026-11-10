"""Test doubles: no network, no cluster, no git/dvc binaries needed."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import requests

MD5 = "a" * 32
COMMIT = "1234567890abcdef1234567890abcdef12345678"


# ----------------------------------------------------------------------------- repo + dvc
def make_repo(root: Path, md5: str = MD5, commit: str = COMMIT) -> Path:
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "refs" / "heads" / "main").write_text(commit + "\n")
    (root / "data").mkdir()
    (root / "data" / "train.csv.dvc").write_text(f"outs:\n- md5: {md5}\n  size: 1000\n  path: train.csv\n")
    return root


def make_runner(table: dict[tuple[str, str], tuple[int, str, str]] | None = None):
    """Fake subprocess.run: keyed by (executable, subcommand) -> (rc, stdout, stderr)."""
    table = table or {}
    defaults = {
        ("git", "status"): (0, "", ""),
        ("dvc", "status"): (0, "{}", ""),
        ("git", "log"): (0, f"{COMMIT}\t2026-10-01T10:00:00+00:00\tTrack dataset v1\n", ""),
    }
    merged = {**defaults, **table}
    calls: list[list[str]] = []

    def runner(args, **kwargs):
        calls.append(list(args))
        rc, out, err = merged.get(tuple(args[:2]), (1, "", "unmocked command"))
        return subprocess.CompletedProcess(args, rc, out, err)

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


# ----------------------------------------------------------------------------- mlflow
def make_run(run_id: str, f1: float, *, acc: float = 0.9, md5: str | None = MD5, start: int = 1, gate: str | None = "true",
             status: str = "FINISHED", experiment_id: str = "1") -> Any:
    tags = {"git_commit": COMMIT, "dvc_file": "data/train.csv.dvc"}
    if md5:
        tags["dvc_md5"] = md5
    if gate:
        tags["quality_gate_passed"] = gate
    return NS(
        info=NS(run_id=run_id, run_name=f"run-{run_id}", experiment_id=experiment_id, status=status, start_time=start),
        data=NS(params={"n_estimators": "100"}, metrics={"f1_macro": f1, "accuracy": acc}, tags=tags),
    )


class FakeMlflowClient:
    def __init__(self, runs: list[Any], models: list[Any] | None = None) -> None:
        self.runs = runs
        self.exp = NS(experiment_id="1", name="mlops-agent-classifier", lifecycle_stage="active")
        self.models = models or []

    def get_experiment_by_name(self, name):
        return self.exp if name == self.exp.name else None

    def get_experiment(self, exp_id):
        if exp_id == self.exp.experiment_id:
            return self.exp
        raise RuntimeError("RESOURCE_DOES_NOT_EXIST")

    def search_experiments(self, max_results=None):
        return [self.exp]

    def search_runs(self, experiment_ids, filter_string=None, order_by=None, max_results=None):
        runs = [r for r in self.runs if r.info.experiment_id in experiment_ids]
        if filter_string and "FINISHED" in filter_string:
            runs = [r for r in runs if r.info.status == "FINISHED"]
        return sorted(runs, key=lambda r: r.info.start_time, reverse=True)[:max_results]

    def get_run(self, run_id):
        for r in self.runs:
            if r.info.run_id == run_id:
                return r
        raise RuntimeError("RESOURCE_DOES_NOT_EXIST")

    def list_artifacts(self, run_id):
        return [NS(path="model_file")]

    def search_registered_models(self, filter_string=None, max_results=None):
        return self.models

    def search_model_versions(self, flt):
        return [NS(version="1", run_id="r1", status="READY", aliases=["candidate"])]


# ----------------------------------------------------------------------------- airflow
class FakeResponse:
    def __init__(self, status: int = 200, body: Any = None, text: str = "") -> None:
        self.status_code = status
        self._body = body
        self.text = text or (str(body) if body is not None else "")

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, routes: dict[tuple[str, str], Any] | None = None, base: str = "http://airflow.test") -> None:
        self.routes = {("POST", "/auth/token"): FakeResponse(200, {"access_token": "jwt.token.value"}), **(routes or {})}
        self.base = base
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, url, headers=None, timeout=None, **kw):
        path = url.replace(self.base, "")
        self.calls.append((method, path, {"headers": headers or {}, **kw}))
        route = self.routes.get((method, path))
        if route is None:
            return FakeResponse(404, {"detail": "not found"})
        if isinstance(route, Exception):
            raise route
        return route(kw) if callable(route) else route

    def posted(self, path: str) -> list[dict]:
        return [c[2] for c in self.calls if c[0] == "POST" and c[1] == path]


class DownSession:
    def request(self, *a, **kw):
        raise requests.exceptions.ConnectionError("connection refused")

    post = request


DAG = "/api/v2/dags/training_pipeline"


def airflow_routes(*, latest_state: str = "success", run_id: str = "manual__1", failed_task: str | None = None):
    run = {"dag_run_id": run_id, "state": latest_state, "run_type": "manual", "logical_date": None,
           "start_date": "2026-10-01T10:00:00Z", "end_date": "2026-10-01T10:05:00Z"}
    tasks = [{"task_id": t, "state": "success", "try_number": 1, "max_tries": 2} for t in
             ("validate_dataset", "preprocess_data", "train_model", "evaluate_and_log", "apply_quality_gate")]
    if failed_task:
        for t in tasks:
            if t["task_id"] == failed_task:
                t["state"] = "failed"
    return {
        ("GET", f"{DAG}/dagRuns"): FakeResponse(200, {"dag_runs": [run]}),
        ("GET", f"{DAG}/dagRuns/{run_id}"): FakeResponse(200, run),
        ("GET", f"{DAG}/dagRuns/{run_id}/taskInstances"): FakeResponse(200, {"task_instances": tasks}),
        ("GET", f"{DAG}/dagRuns/{run_id}/taskInstances/apply_quality_gate/logs/1"):
            FakeResponse(200, {"content": ["gate started", "f1_macro=0.7000 is below threshold 0.80"]}),
        ("GET", "/api/v2/dags"): FakeResponse(200, {"dags": [{"dag_id": "training_pipeline", "is_paused": False, "tags": []}]}),
        ("GET", DAG): FakeResponse(200, {"dag_id": "training_pipeline", "is_paused": False}),
        ("POST", f"{DAG}/dagRuns"): FakeResponse(200, {"dag_run_id": "manual__new", "state": "queued"}),
        ("GET", f"{DAG}/dagRuns/manual__new"): FakeResponse(200, {"dag_run_id": "manual__new", "state": "success"}),
    }


# ----------------------------------------------------------------------------- kubernetes
def make_pod(name: str, *, phase="Running", ready=True, restarts=0, waiting=None, terminated=None, image="mlops-inference:0.1.0"):
    cs = NS(
        ready=ready,
        restart_count=restarts,
        state=NS(waiting=NS(reason=waiting) if waiting else None),
        last_state=NS(terminated=NS(reason=terminated) if terminated else None),
    )
    return NS(
        metadata=NS(name=name, labels={"app": "mlops-inference"}),
        status=NS(phase=phase, container_statuses=[cs]),
        spec=NS(node_name="node-1", containers=[NS(image=image)]),
    )


def make_deployment(*, desired=2, ready=2, image="mlops-inference:0.1.0", version="0.1.0"):
    return NS(
        metadata=NS(name="mlops-inference", labels={"app.kubernetes.io/version": version}),
        spec=NS(replicas=desired, template=NS(spec=NS(containers=[NS(image=image)]))),
        status=NS(ready_replicas=ready, available_replicas=ready, updated_replicas=desired),
    )


class FakeCore:
    def __init__(self, pods, *, endpoints=2, events=None, app_version="0.1.0") -> None:
        self.pods, self.endpoints, self.events, self.app_version = pods, endpoints, events or [], app_version
        self.log_calls: list[dict] = []

    def list_namespaced_pod(self, ns, label_selector=None, **kw):
        return NS(items=self.pods)

    def list_namespaced_service(self, ns, label_selector=None, **kw):
        svc = NS(metadata=NS(name="mlops-inference"),
                 spec=NS(type="ClusterIP", cluster_ip="10.0.0.1", selector={"app": "mlops-inference"},
                         ports=[NS(port=80, target_port=8000)]))
        return NS(items=[svc])

    def read_namespaced_endpoints(self, name, ns, **kw):
        return NS(subsets=[NS(addresses=[object()] * self.endpoints)] if self.endpoints else [])

    def read_namespaced_config_map(self, name, ns, **kw):
        return NS(data={"APP_VERSION": self.app_version, "MODEL_RUN_ID": "r-best"})

    def read_namespaced_pod_log(self, name, ns, tail_lines=None, previous=False, **kw):
        self.log_calls.append({"pod": name, "previous": previous})
        return "Traceback ... token=abc123secret\nboom"

    def list_namespaced_event(self, ns, field_selector=None, **kw):
        return NS(items=self.events)


class FakeApps:
    def __init__(self, deployment) -> None:
        self.deployment = deployment
        self.patches: list[tuple] = []

    def list_namespaced_deployment(self, ns, label_selector=None, **kw):
        return NS(items=[self.deployment])

    def read_namespaced_deployment(self, name, ns, **kw):
        return self.deployment

    def patch_namespaced_deployment(self, name, ns, body, **kw):
        self.patches.append((name, ns, body))


def k8s_factory(pods=None, deployment=None, **core_kw):
    from agent.tools.kubernetes_tools import K8sClients

    pods = pods if pods is not None else [make_pod("p1"), make_pod("p2")]
    clients = K8sClients(core=FakeCore(pods, **core_kw), apps=FakeApps(deployment or make_deployment()))
    return (lambda: clients), clients
