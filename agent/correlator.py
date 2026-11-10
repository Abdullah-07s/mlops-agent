"""Turns raw tool results into: known facts, uncertainties and recommended actions.

All statements here are derived deterministically from tool output. The LLM may only
rephrase them (and its text is rejected if it introduces numbers that are not in the
evidence), so the agent cannot invent operational results.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from agent.config import Settings
from agent.schemas import ChatRequest, Fact, ToolResult, ToolStatus

Summarizer = Callable[[dict[str, Any]], list[str]]


def _short(value: Any, n: int = 8) -> str:
    return str(value)[:n] if value else "unknown"


def _best_run(d: dict[str, Any]) -> list[str]:
    best = d["best"]
    out = [
        f"Best run in '{d['experiment']['name']}' by {d['metric']} ({d['mode']}): {best['run_id']} "
        f"({best.get('run_name')}) = {d['best_value']:.4f}; params={best['params']}; "
        f"{d['considered']} finished run(s) considered."
    ]
    prov = best["provenance"]
    out.append(
        f"Run {_short(best['run_id'])} provenance tags: dvc_md5={_short(prov.get('dvc_md5'), 12)}, "
        f"git_commit={_short(prov.get('git_commit'))}, quality_gate_passed={best.get('quality_gate_passed')}."
    )
    return out


def _compare(d: dict[str, Any]) -> list[str]:
    table = ", ".join(f"{_short(r['run_id'])}={r['value']:.4f}" for r in d["table"])
    lead = f"; leader {d['winner']['run_id']} ({d['winner_value']:.4f})" if d.get("winner") else ""
    return [f"Compared {len(d['runs'])} runs on {d['metric']}: {table or 'no values'}{lead}."]


def _list_runs(d: dict[str, Any]) -> list[str]:
    runs = d["runs"]
    latest = f"; latest {runs[0]['run_id']} status={runs[0]['status']}" if runs else ""
    return [f"Experiment '{d['experiment']['name']}' has {len(runs)} recent run(s){latest}."]


def _versions(d: dict[str, Any]) -> list[str]:
    out = []
    for ds in d["datasets"]:
        out.append(f"Dataset {ds['path']} (from {ds['dvc_file']}) has DVC hash {ds['md5']}.")
    dirty = {True: "dirty", False: "clean", None: "unknown"}[d["git_dirty"]]
    return [f"Repository is at git commit {d['git_commit']} (working tree {dirty})."] + out


def _find_commit(d: dict[str, Any]) -> list[str]:
    if d["found"]:
        return [f"Dataset hash {d['md5'][:12]} was introduced in git commit {d['introducing_commit']}."]
    return [f"No commit in the history of {d['dvc_file']} contains hash {d['md5'][:12]}."]


def _unhealthy(d: dict[str, Any]) -> list[str]:
    if not d["unhealthy"]:
        return [f"All {d['total']} pod(s) in namespace '{d['namespace']}' are healthy."]
    return [f"Pod {p['name']} unhealthy: {', '.join(p['problems'])}." for p in d["unhealthy"]]


def _version(d: dict[str, Any]) -> list[str]:
    out = [
        f"Deployment '{d['deployment']}' runs image {d['image']} "
        f"(ready {d['replicas']['ready']}/{d['replicas']['desired']}); version label={d['version_label']}, "
        f"ConfigMap APP_VERSION={d['configmap_app_version']}."
    ]
    seen = {v for v in (d["image_tag"], d["version_label"], d["configmap_app_version"]) if v}
    if len(seen) > 1:
        out.append(f"Version sources disagree: {sorted(seen)}.")
    return out


def _dag_runs(d: dict[str, Any]) -> list[str]:
    if not d["runs"]:
        return [f"DAG '{d['dag_id']}' has no runs."]
    runs = ", ".join(f"{r['dag_run_id']}={r['state']}" for r in d["runs"][:3])
    return [f"Recent runs of '{d['dag_id']}': {runs}; latest state={d['latest_state']}."]


def _failed_tasks(d: dict[str, Any]) -> list[str]:
    failed = [t["task_id"] for t in d["failed"]]
    if not failed:
        return [f"Run {d['dag_run_id']} (state={d['dag_run_state']}) has no failed tasks."]
    return [f"Run {d['dag_run_id']} (state={d['dag_run_state']}) failed task(s): {', '.join(failed)}."]


SUMMARIZERS: dict[str, Summarizer] = {
    "mlflow_best_run": _best_run,
    "mlflow_compare_runs": _compare,
    "mlflow_list_runs": _list_runs,
    "mlflow_list_experiments": lambda d: [f"{d['count']} experiment(s): {', '.join(e['name'] for e in d['experiments'])}."],
    "mlflow_get_run": lambda d: [f"Run {d['run']['run_id']} status={d['run']['status']}, metrics={d['run']['metrics']}."],
    "mlflow_get_registry_info": lambda d: [f"{d['count']} registered model(s): " + "; ".join(f"{m['name']} ({len(m['versions'])} versions)" for m in d["models"]) + "."],
    "dvc_list_datasets": lambda d: [f"{d['count']} DVC-tracked dataset(s): " + ", ".join(f"{x['path']}@{x['version_id']}" for x in d["datasets"]) + "."],
    "dvc_current_version": _versions,
    "dvc_compare_versions": lambda d: [f"Dataset in {d['dvc_file']} {'DIFFERS' if d['changed'] else 'is identical'} between {d['a']['rev']} and {d['b']['rev']} (size delta {d['size_delta_bytes']} bytes)."],
    "dvc_detect_changes": lambda d: [f"DVC status: tracked data/pipelines {'CHANGED' if d['changed'] else 'up to date'}."],
    "dvc_find_commit_for_hash": _find_commit,
    "dvc_reproduce_steps": lambda d: [f"Reproduction plan for dataset at {d['target']['commit'][:8]} has {len(d['steps'])} steps (see tool trace)."],
    "k8s_list_deployments": lambda d: [f"Deployment {x['name']}: {x['ready']}/{x['desired']} ready, image {x['image']}." for x in d["deployments"]] or ["No matching Deployments found."],
    "k8s_list_pods": lambda d: [f"{len(d['pods'])} pod(s): " + ", ".join(f"{p['name']}={p['phase']}{'' if p['ready'] else '/not-ready'}" for p in d["pods"]) + "."],
    "k8s_get_pod_logs": lambda d: [f"Retrieved log tail of pod {d['pod']} ({len(d['log_tail'])} chars)."],
    "k8s_list_services": lambda d: [f"Service {s['name']} ({s['type']}) has {s['ready_endpoints']} ready endpoint(s)." for s in d["services"]] or ["No matching Services found."],
    "k8s_find_unhealthy_pods": _unhealthy,
    "k8s_get_deployed_version": _version,
    "k8s_diagnose_deployment": lambda d: [f"{f['problem']} (evidence: {f['evidence']})" for f in d["findings"]] or ["Diagnosis found no problems in deployment, pods, events or endpoints."],
    "k8s_restart_deployment": lambda d: [f"Rolling restart of '{d['deployment']}' requested in '{d['namespace']}'."],
    "airflow_list_dags": lambda d: [f"{d['count']} DAG(s): {', '.join(x['dag_id'] for x in d['dags'])}; approved for triggering: {d['approved']}."],
    "airflow_get_dag_runs": _dag_runs,
    "airflow_trigger_training_dag": lambda d: [f"Triggered '{d['dag_id']}' run {d['dag_run_id']}; state={d['state']} (finished={d['finished']})."],
    "airflow_get_task_instances": lambda d: [f"Run {d['dag_run_id']} tasks: " + ", ".join(f"{t['task_id']}={t['state']}" for t in d["tasks"]) + "."],
    "airflow_get_failed_tasks": _failed_tasks,
    "airflow_get_task_logs": lambda d: [f"Retrieved log tail for task {d['task_id']} attempt {d['try_number']} ({d['lines_returned']} lines)."],
}


def _last_ok(history: list[ToolResult], tool: str) -> dict[str, Any] | None:
    for r in reversed(history):
        if r.tool == tool and r.ok:
            return r.data
    return None


def facts_from(history: list[ToolResult]) -> list[Fact]:
    facts: list[Fact] = []
    for r in history:
        if not r.ok:
            continue
        summarizer = SUMMARIZERS.get(r.tool)
        try:
            statements = summarizer(r.data) if summarizer else [f"{r.tool} returned {sorted(r.data)}."]
        except (KeyError, TypeError, IndexError):
            statements = [f"{r.tool} returned data in an unexpected shape."]
        facts += [Fact(source=r.tool, statement=s) for s in statements]
    return facts


def uncertainties_from(history: list[ToolResult]) -> list[str]:
    out = []
    for r in history:
        if r.ok:
            continue
        label = {
            ToolStatus.TIMEOUT: "timed out",
            ToolStatus.REJECTED: "was blocked by a safety control",
            ToolStatus.INVALID: "received an invalid request",
            ToolStatus.ERROR: "failed",
        }[r.status]
        out.append(f"{r.tool} {label} ({r.error_code}): {r.error} - claims that depend on it are unverified.")
    return out


def recommend(
    history: list[ToolResult], request: ChatRequest, settings: Settings
) -> tuple[list[str], list[str]]:
    """Returns (recommendations, extra_uncertainties). Retraining is only ever recommended."""
    recs: list[str] = []
    unc: list[str] = []
    best = _last_ok(history, "mlflow_best_run")
    runs = _last_ok(history, "airflow_get_dag_runs")
    failed = _last_ok(history, "airflow_get_failed_tasks")

    reasons: list[str] = []
    if failed and any(t["task_id"] == "apply_quality_gate" for t in failed["failed"]):
        reasons.append(f"run {failed['dag_run_id']} failed the apply_quality_gate task")
    if best and settings.min_metric_threshold is not None and best["metric"] == settings.default_metric:
        if best["best_value"] < settings.min_metric_threshold:
            reasons.append(
                f"best {best['metric']}={best['best_value']:.4f} is below the policy threshold "
                f"{settings.min_metric_threshold}"
            )
    if best and settings.freshness_days is not None:
        age_days = (time.time() * 1000 - best["best"]["start_time"]) / 86_400_000
        if age_days > settings.freshness_days:
            reasons.append(f"best model is {age_days:.0f} days old (policy: {settings.freshness_days})")
    elif best:
        unc.append("No freshness policy is configured (FRESHNESS_DAYS unset); model age is not used to justify retraining.")
    if request.degradation_note:
        reasons.append(f"user-supplied degradation report: {request.degradation_note}")

    if reasons:
        recs.append("Retraining may be warranted (NOT started automatically): " + "; ".join(reasons) + ".")
    elif best or runs:
        recs.append("No retraining trigger found in the collected evidence (no failed quality gate, no metric below policy, no reported degradation).")

    if best:
        recs.append("Do not treat the top metric as production approval: verify latency, quality gate and human review first.")
    unhealthy = _last_ok(history, "k8s_find_unhealthy_pods")
    if unhealthy and unhealthy["unhealthy"]:
        recs.append("Investigate unhealthy pods with k8s_diagnose_deployment; restarting requires explicit confirmation.")
    if runs and runs["latest_state"] == "failed" and not failed:
        recs.append("Latest training run failed: ask for its failed tasks and logs.")
    if failed and failed["failed"]:
        recs.append("Inspect the failed task logs before re-running the pipeline.")
    changes = _last_ok(history, "dvc_detect_changes")
    if changes and changes["changed"]:
        recs.append("Tracked data/pipeline changed since the last commit: review 'dvc status' and commit or revert before training.")
    commit = _last_ok(history, "dvc_find_commit_for_hash")
    if commit and not commit["found"]:
        unc.append("The run's dataset hash is not in git history: it may have been trained on uncommitted data.")
    return recs, unc
