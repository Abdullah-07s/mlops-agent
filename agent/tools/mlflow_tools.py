"""Task 2 - MLflow experiment intelligence tools (MlflowClient only, no UI scraping)."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.config import Settings
from agent.errors import ToolError
from agent.registry import ToolSpec

_ID = r"^[A-Za-z0-9_\-]{4,64}$"
_NAME = r"^[A-Za-z0-9_.\- ]{1,128}$"


class ListExperimentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(20, ge=1, le=100)


class ListRunsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment: str | None = Field(None, max_length=128)
    limit: int = Field(10, ge=1, le=100)


class GetRunArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(pattern=_ID)


class BestRunArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment: str | None = Field(None, max_length=128)
    metric: str | None = Field(None, pattern=_NAME)
    mode: Literal["max", "min"] = "max"
    recent: int = Field(50, ge=1, le=500)


class CompareRunsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_ids: list[str] = Field(default_factory=list, max_length=10)
    experiment: str | None = Field(None, max_length=128)
    last_n: int = Field(5, ge=2, le=20)
    metric: str | None = Field(None, pattern=_NAME)
    mode: Literal["max", "min"] = "max"


class RegistryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    registered_model: str | None = Field(None, pattern=_NAME)


def default_client_factory(settings: Settings) -> Callable[[], Any]:
    def factory() -> Any:
        # Fail fast instead of MLflow's default 7 retries with exponential backoff.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
        try:
            from mlflow import MlflowClient
        except ImportError as exc:
            raise ToolError("dependency_missing", "mlflow is not installed: pip install -e '.[mlflow]'") from exc
        return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    return factory


def summarize_run(run: Any) -> dict[str, Any]:
    info, data = run.info, run.data
    tags = dict(data.tags)
    return {
        "run_id": info.run_id,
        "run_name": getattr(info, "run_name", None) or tags.get("mlflow.runName"),
        "experiment_id": info.experiment_id,
        "status": info.status,
        "start_time": info.start_time,
        "params": dict(data.params),
        "metrics": dict(data.metrics),
        "provenance": {
            "dvc_md5": tags.get("dvc_md5"),
            "git_commit": tags.get("git_commit"),
            "dvc_file": tags.get("dvc_file"),
        },
        "quality_gate_passed": tags.get("quality_gate_passed"),
    }


def _caveats(run: dict[str, Any]) -> list[str]:
    notes = ["Metric rank alone does not establish production readiness (latency, fairness, review not evaluated)."]
    if run.get("quality_gate_passed") != "true":
        notes.append(f"Run has no passing quality-gate tag (quality_gate_passed={run.get('quality_gate_passed')}).")
    if not run["provenance"].get("dvc_md5"):
        notes.append("Run has no dvc_md5 tag, so its dataset version cannot be established from MLflow.")
    return notes

def _tie_caveat(scored: list[tuple[float, Any]], metric: str) -> list[str]:
    if len(scored) < 2 or scored[0][0] != scored[1][0]:
        return []
    n = sum(1 for value, _ in scored if value == scored[0][0])
    return [
        f"{n} runs tie at {scored[0][0]:.4f} on {metric}; the leader was chosen by recency "
        "(newest first), not by a margin."
    ]

def build_tools(settings: Settings, client_factory: Callable[[], Any] | None = None) -> list[ToolSpec]:
    factory = client_factory or default_client_factory(settings)
    cache: dict[str, Any] = {}

    def client() -> Any:
        if "c" not in cache:
            cache["c"] = factory()
        return cache["c"]

    def resolve_experiment(ref: str | None) -> Any:
        ref = ref or settings.default_experiment
        try:
            exp = client().get_experiment(ref) if ref.isdigit() else client().get_experiment_by_name(ref)
        except ToolError:
            raise
        except Exception:  # noqa: BLE001 - MLflow raises MlflowException for unknown ids
            exp = None
        if exp is None or getattr(exp, "lifecycle_stage", "active") == "deleted":
            raise ToolError("experiment_not_found", f"Experiment '{ref}' not found")
        return exp

    def recent_finished(exp_id: str, n: int) -> list[Any]:
        return list(
            client().search_runs(
                [exp_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["attributes.start_time DESC"],
                max_results=n,
            )
        )

    def rank(runs: list[Any], metric: str, mode: str) -> list[tuple[float, Any]]:
        scored = [(r.data.metrics[metric], r) for r in runs if metric in r.data.metrics]
        scored.sort(key=lambda t: t[0], reverse=(mode == "max"))  # stable: ties keep newest first
        return scored

    def list_experiments(a: ListExperimentsArgs) -> dict[str, Any]:
        exps = client().search_experiments(max_results=a.limit)
        items = [{"experiment_id": e.experiment_id, "name": e.name} for e in exps]
        return {"experiments": items, "count": len(items)}

    def list_runs(a: ListRunsArgs) -> dict[str, Any]:
        exp = resolve_experiment(a.experiment)
        runs = client().search_runs(
            [exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=a.limit
        )
        return {
            "experiment": {"experiment_id": exp.experiment_id, "name": exp.name},
            "runs": [summarize_run(r) for r in runs],
        }

    def get_run(a: GetRunArgs) -> dict[str, Any]:
        try:
            run = client().get_run(a.run_id)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError("run_not_found", f"Run {a.run_id} not found: {str(exc)[:120]}") from exc
        try:
            artifacts = [f.path for f in client().list_artifacts(a.run_id)]
        except Exception:  # noqa: BLE001 - artifact store may be unreachable; metadata is still valid
            artifacts = None
        return {"run": summarize_run(run), "artifacts": artifacts}

    def best_run(a: BestRunArgs) -> dict[str, Any]:
        metric = a.metric or settings.default_metric
        exp = resolve_experiment(a.experiment)
        runs = recent_finished(exp.experiment_id, a.recent)
        scored = rank(runs, metric, a.mode)
        if not scored:
            raise ToolError(
                "no_runs_with_metric",
                f"No FINISHED runs in '{exp.name}' log metric '{metric}' (considered {len(runs)} runs)",
            )
        best = summarize_run(scored[0][1])
        return {
            "experiment": {"experiment_id": exp.experiment_id, "name": exp.name},
            "metric": metric,
            "mode": a.mode,
            "considered": len(runs),
            "best_value": scored[0][0],
            "best": best,
            "ranking": [{"run_id": r.info.run_id, "value": v} for v, r in scored[:5]],
            "caveats": _caveats(best) + _tie_caveat(scored, metric),    
        }

    def compare_runs(a: CompareRunsArgs) -> dict[str, Any]:
        metric = a.metric or settings.default_metric
        if a.run_ids:
            runs = []
            for rid in a.run_ids:
                try:
                    runs.append(client().get_run(rid))
                except Exception as exc:  # noqa: BLE001
                    raise ToolError("run_not_found", f"Run {rid} not found: {str(exc)[:120]}") from exc
        else:
            exp = resolve_experiment(a.experiment)
            runs = recent_finished(exp.experiment_id, a.last_n)
        if len(runs) < 2:
            raise ToolError("not_enough_runs", f"Need at least 2 runs to compare, found {len(runs)}")
        scored = rank(runs, metric, a.mode)
        summaries = [summarize_run(r) for r in runs]
        winner = summarize_run(scored[0][1]) if scored else None
        return {
            "metric": metric,
            "mode": a.mode,
            "runs": summaries,
            "table": [{"run_id": r.info.run_id, "value": v} for v, r in scored],
            "winner": winner,
            "winner_value": scored[0][0] if scored else None,
            "caveats": (_caveats(winner) + _tie_caveat(scored, metric))
            if winner
            else [f"No compared run logged metric '{metric}'."],
        }

    def registry_info(a: RegistryArgs) -> dict[str, Any]:
        flt = f"name = '{a.registered_model}'" if a.registered_model else None
        try:
            models = client().search_registered_models(filter_string=flt, max_results=50)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError("registry_unavailable", f"Model registry not available: {str(exc)[:150]}") from exc
        out = []
        for m in models:
            versions = client().search_model_versions(f"name = '{m.name}'")
            out.append(
                {
                    "name": m.name,
                    "aliases": dict(getattr(m, "aliases", {}) or {}),
                    "versions": [
                        {
                            "version": v.version,
                            "run_id": getattr(v, "run_id", None),
                            "status": getattr(v, "status", None),
                            "aliases": list(getattr(v, "aliases", []) or []),
                        }
                        for v in versions
                    ],
                }
            )
        return {"models": out, "count": len(out)}

    return [
        ToolSpec("mlflow_list_experiments", "List MLflow experiments.", ListExperimentsArgs, list_experiments),
        ToolSpec("mlflow_list_runs", "List recent runs of an experiment (name or numeric id) with params, metrics and provenance tags.", ListRunsArgs, list_runs),
        ToolSpec("mlflow_get_run", "Get params, metrics, tags and artifacts of one run.", GetRunArgs, get_run),
        ToolSpec("mlflow_best_run", "Best FINISHED run for a metric (default f1_macro) with ranking and caveats.", BestRunArgs, best_run),
        ToolSpec("mlflow_compare_runs", "Compare the last N runs (or given run ids) on a metric.", CompareRunsArgs, compare_runs),
        ToolSpec("mlflow_get_registry_info", "Registered models, versions and aliases when a registry is available.", RegistryArgs, registry_info),
    ]
