"""Task 3 - Apache Airflow 3 workflow tools (REST API v2 with JWT bearer auth)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import requests
from pydantic import BaseModel, ConfigDict, Field

from agent.config import Settings
from agent.errors import ToolError
from agent.logging_utils import redact_text
from agent.registry import ToolSpec

_DAG = r"^[A-Za-z0-9_.\-]{1,250}$"
_RUN = r"^[A-Za-z0-9_.:+\-]{1,250}$"
_TASK = r"^[A-Za-z0-9_.\-]{1,250}$"
TERMINAL_STATES = {"success", "failed"}


class AirflowClient:
    """Minimal authenticated client: POST /auth/token, then Bearer on /api/v2/*."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        *,
        session: Any | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._session = session or requests.Session()
        self._timeout = timeout
        self._token: str | None = None

    def _send(self, method: str, path: str, *, auth: bool = True, **kwargs: Any) -> Any:
        for attempt in (1, 2):
            headers = dict(kwargs.pop("headers", {}) or {})
            if auth:
                headers["Authorization"] = f"Bearer {self._token or self._login()}"
            try:
                resp = self._session.request(
                    method, f"{self.base_url}{path}", headers=headers, timeout=self._timeout, **kwargs
                )
            except requests.exceptions.Timeout as exc:
                raise ToolError("airflow_timeout", f"Airflow did not answer within {self._timeout}s") from exc
            except requests.exceptions.RequestException as exc:
                raise ToolError("airflow_unreachable", redact_text(f"Cannot reach Airflow: {exc}")[:250]) from exc
            if resp.status_code == 401 and auth and attempt == 1:
                self._token = None  # token expired: log in again once
                continue
            if resp.status_code >= 400:
                raise ToolError(
                    f"airflow_http_{resp.status_code}",
                    redact_text(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}"),
                )
            return resp
        raise ToolError("airflow_auth_failed", "Airflow rejected the credentials")

    def _login(self) -> str:
        if not self._password:
            raise ToolError("airflow_not_configured", "AIRFLOW_PASSWORD is not set")
        resp = self._send(
            "POST",
            "/auth/token",
            auth=False,
            json={"username": self._username, "password": self._password},
        )
        token = resp.json().get("access_token")
        if not token:
            raise ToolError("airflow_auth_failed", "Airflow returned no access_token")
        self._token = token
        return token

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._send("GET", path, params=params).json()

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        return self._send("POST", path, json=payload).json()

    def get_raw(self, path: str, accept: str) -> Any:
        return self._send("GET", path, headers={"Accept": accept})


class DagArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dag_id: str = Field("training_pipeline", pattern=_DAG)


class ListDagsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(50, ge=1, le=100)


class DagRunsArgs(DagArgs):
    limit: int = Field(5, ge=1, le=50)


class RunArgs(DagArgs):
    dag_run_id: str | None = Field(None, pattern=_RUN)


class TriggerArgs(DagArgs):
    conf: dict[str, Any] = Field(default_factory=dict)
    wait_seconds: int = Field(0, ge=0, le=120)


class TaskLogArgs(DagArgs):
    dag_run_id: str = Field(pattern=_RUN)
    task_id: str = Field(pattern=_TASK)
    try_number: int = Field(1, ge=1, le=20)


def _run_fields(run: dict[str, Any]) -> dict[str, Any]:
    keys = ("dag_run_id", "state", "run_type", "logical_date", "start_date", "end_date")
    return {k: run.get(k) for k in keys}


def _ti_fields(ti: dict[str, Any]) -> dict[str, Any]:
    keys = ("task_id", "state", "try_number", "max_tries", "start_date", "end_date", "duration")
    return {k: ti.get(k) for k in keys}


def build_tools(
    settings: Settings,
    client: AirflowClient | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[ToolSpec]:
    holder: dict[str, AirflowClient] = {}

    def c() -> AirflowClient:
        if "c" not in holder:
            holder["c"] = client or AirflowClient(
                settings.airflow_base_url, settings.airflow_username, settings.airflow_password
            )
        return holder["c"]

    def run_path(dag_id: str, run_id: str) -> str:
        return f"/api/v2/dags/{dag_id}/dagRuns/{quote(run_id, safe='')}"

    def resolve_run(dag_id: str, run_id: str | None) -> dict[str, Any]:
        if run_id:
            return c().get_json(run_path(dag_id, run_id))
        runs = c().get_json(
            f"/api/v2/dags/{dag_id}/dagRuns", params={"limit": 1, "order_by": "-start_date"}
        ).get("dag_runs", [])
        if not runs:
            raise ToolError("no_dag_runs", f"DAG '{dag_id}' has no runs yet")
        return runs[0]

    def task_instances(dag_id: str, run_id: str) -> list[dict[str, Any]]:
        body = c().get_json(f"{run_path(dag_id, run_id)}/taskInstances")
        return [_ti_fields(t) for t in body.get("task_instances", [])]

    def list_dags(a: ListDagsArgs) -> dict[str, Any]:
        body = c().get_json("/api/v2/dags", params={"limit": a.limit})
        dags = [
            {"dag_id": d.get("dag_id"), "is_paused": d.get("is_paused"), "tags": [t.get("name") for t in d.get("tags", [])]}
            for d in body.get("dags", [])
        ]
        return {"dags": dags, "count": len(dags), "approved": sorted(settings.approved_dags)}

    def get_dag_runs(a: DagRunsArgs) -> dict[str, Any]:
        body = c().get_json(
            f"/api/v2/dags/{a.dag_id}/dagRuns", params={"limit": a.limit, "order_by": "-start_date"}
        )
        runs = [_run_fields(r) for r in body.get("dag_runs", [])]
        return {"dag_id": a.dag_id, "runs": runs, "latest_state": runs[0]["state"] if runs else None}

    def trigger(a: TriggerArgs) -> dict[str, Any]:
        if a.dag_id not in settings.approved_dags:
            raise ToolError(
                "dag_not_approved",
                f"DAG '{a.dag_id}' is not approved for triggering (approved: {sorted(settings.approved_dags)})",
            )
        meta = c().get_json(f"/api/v2/dags/{a.dag_id}")
        run = c().post_json(f"/api/v2/dags/{a.dag_id}/dagRuns", {"logical_date": None, "conf": a.conf})
        run_id, state = run.get("dag_run_id"), run.get("state")
        deadline = time.monotonic() + a.wait_seconds
        while a.wait_seconds and state not in TERMINAL_STATES and time.monotonic() < deadline:
            sleep(5)
            state = c().get_json(run_path(a.dag_id, run_id)).get("state")
        out = {
            "dag_id": a.dag_id,
            "dag_run_id": run_id,
            "state": state,
            "finished": state in TERMINAL_STATES,
            "dag_paused": bool(meta.get("is_paused")),
        }
        if out["dag_paused"]:
            out["note"] = "DAG is paused; the run may stay queued until the DAG is unpaused."
        return out

    def get_task_instances(a: RunArgs) -> dict[str, Any]:
        run = resolve_run(a.dag_id, a.dag_run_id)
        tis = task_instances(a.dag_id, run["dag_run_id"])
        return {"dag_id": a.dag_id, "dag_run_id": run["dag_run_id"], "state": run.get("state"), "tasks": tis}

    def get_failed_tasks(a: RunArgs) -> dict[str, Any]:
        run = resolve_run(a.dag_id, a.dag_run_id)
        tis = task_instances(a.dag_id, run["dag_run_id"])
        return {
            "dag_id": a.dag_id,
            "dag_run_id": run["dag_run_id"],
            "dag_run_state": run.get("state"),
            "task_count": len(tis),
            "failed": [t for t in tis if t["state"] == "failed"],
            "upstream_failed": [t for t in tis if t["state"] == "upstream_failed"],
        }

    def get_task_logs(a: TaskLogArgs) -> dict[str, Any]:
        path = f"{run_path(a.dag_id, a.dag_run_id)}/taskInstances/{a.task_id}/logs/{a.try_number}"
        resp = c().get_raw(path, "application/json")
        try:
            content = resp.json().get("content", [])
            lines = [x if isinstance(x, str) else str(x.get("event", x)) for x in content]
        except ValueError:
            lines = resp.text.splitlines()
        tail = redact_text("\n".join(lines[-60:]))[-4000:]
        return {"task_id": a.task_id, "try_number": a.try_number, "log_tail": tail, "lines_returned": len(lines)}

    return [
        ToolSpec("airflow_list_dags", "List Airflow DAGs and which ones are approved for triggering.", ListDagsArgs, list_dags),
        ToolSpec("airflow_get_dag_runs", "Recent DAG runs and their states.", DagRunsArgs, get_dag_runs),
        ToolSpec("airflow_trigger_training_dag", "Trigger an approved training DAG (changes state; needs confirmation).", TriggerArgs, trigger, mutating=True, timeout_s=150.0),
        ToolSpec("airflow_get_task_instances", "Task states for a DAG run (default: latest run).", RunArgs, get_task_instances),
        ToolSpec("airflow_get_failed_tasks", "Failed / upstream_failed tasks for a DAG run (default: latest run).", RunArgs, get_failed_tasks),
        ToolSpec("airflow_get_task_logs", "Tail of a task attempt's log for failure diagnosis.", TaskLogArgs, get_task_logs),
    ]
