"""Task 4 - Kubernetes operations tools, restricted to approved namespaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.config import Settings
from agent.errors import ToolError
from agent.logging_utils import redact_text
from agent.registry import ToolSpec

_NS = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"
_NAME = r"^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$"
_TIMEOUT = {"_request_timeout": 10}
BAD_WAITING = {
    "CrashLoopBackOff",
    "ImagePullBackOff",
    "ErrImagePull",
    "CreateContainerConfigError",
    "CreateContainerError",
    "InvalidImageName",
    "RunContainerError",
}


@dataclass
class K8sClients:
    core: Any
    apps: Any


def default_clients() -> K8sClients:
    try:
        from kubernetes import client, config
    except ImportError as exc:
        raise ToolError("dependency_missing", "kubernetes client missing: pip install -e '.[k8s]'") from exc
    try:
        config.load_incluster_config()
    except config.ConfigException:
        try:
            config.load_kube_config()
        except Exception as exc:  # noqa: BLE001
            raise ToolError("k8s_not_configured", f"No usable kubeconfig: {str(exc)[:150]}") from exc
    return K8sClients(core=client.CoreV1Api(), apps=client.AppsV1Api())


def _g(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, part, None)
    return default if cur is None else cur


class NsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    namespace: str | None = Field(None, pattern=_NS)


class DeploymentArgs(NsArgs):
    deployment: str | None = Field(None, pattern=_NAME)


class PodLogsArgs(NsArgs):
    pod_name: str = Field(pattern=_NAME, max_length=253)
    tail_lines: int = Field(100, ge=1, le=500)
    previous: bool = False


def pod_summary(pod: Any) -> dict[str, Any]:
    statuses = _g(pod, "status.container_statuses", []) or []
    waiting = [w for cs in statuses if (w := _g(cs, "state.waiting.reason"))]
    terminated = [t for cs in statuses if (t := _g(cs, "last_state.terminated.reason"))]
    summary = {
        "name": _g(pod, "metadata.name"),
        "phase": _g(pod, "status.phase"),
        "ready": bool(statuses) and all(bool(cs.ready) for cs in statuses),
        "restarts": sum(int(cs.restart_count or 0) for cs in statuses),
        "waiting_reasons": waiting,
        "last_terminated_reasons": terminated,
        "node": _g(pod, "spec.node_name"),
        "images": [c.image for c in (_g(pod, "spec.containers", []) or [])],
    }
    summary["problems"] = pod_problems(summary)
    return summary


def pod_problems(s: dict[str, Any]) -> list[str]:
    problems = []
    if s["phase"] not in ("Running", "Succeeded"):
        problems.append(f"phase={s['phase']}")
    if s["phase"] == "Running" and not s["ready"]:
        problems.append("running but not ready")
    problems += [f"waiting:{w}" for w in s["waiting_reasons"] if w in BAD_WAITING]
    if s["restarts"] > 0:
        problems.append(f"restarts={s['restarts']}")
    problems += [f"last_terminated:{t}" for t in s["last_terminated_reasons"] if t in ("OOMKilled", "Error")]
    return problems


def deployment_summary(d: Any) -> dict[str, Any]:
    containers = _g(d, "spec.template.spec.containers", []) or []
    labels = _g(d, "metadata.labels", {}) or {}
    return {
        "name": _g(d, "metadata.name"),
        "desired": _g(d, "spec.replicas", 0),
        "ready": _g(d, "status.ready_replicas", 0),
        "available": _g(d, "status.available_replicas", 0),
        "updated": _g(d, "status.updated_replicas", 0),
        "image": containers[0].image if containers else None,
        "version_label": labels.get("app.kubernetes.io/version"),
    }


def image_tag(image: str | None) -> str | None:
    if not image:
        return None
    if "@" in image:
        return image.split("@", 1)[1]
    last = image.rsplit("/", 1)[-1]
    return last.split(":", 1)[1] if ":" in last else "latest"


EXPLANATIONS = {
    "ImagePullBackOff": "The image cannot be pulled - check the image name/tag, registry access, or load the image into the cluster.",
    "ErrImagePull": "The image cannot be pulled - check the image name/tag and registry access.",
    "CrashLoopBackOff": "The container keeps crashing - inspect the previous container logs.",
    "CreateContainerConfigError": "Container config is invalid - often a missing ConfigMap/Secret key.",
    "OOMKilled": "The container exceeded its memory limit.",
}


def build_tools(settings: Settings, clients_factory: Callable[[], K8sClients] | None = None) -> list[ToolSpec]:
    holder: dict[str, K8sClients] = {}

    def clients() -> K8sClients:
        if "c" not in holder:
            holder["c"] = (clients_factory or default_clients)()
        return holder["c"]

    def ns_of(namespace: str | None) -> str:
        ns = namespace or settings.k8s_default_namespace
        if ns not in settings.k8s_allowed_namespaces:
            raise ToolError(
                "namespace_not_allowed",
                f"Namespace '{ns}' is not approved (allowed: {sorted(settings.k8s_allowed_namespaces)})",
            )
        return ns

    def pods(ns: str) -> list[dict[str, Any]]:
        items = clients().core.list_namespaced_pod(ns, label_selector=settings.k8s_label_selector, **_TIMEOUT).items
        return [pod_summary(p) for p in items]

    def services(ns: str) -> list[dict[str, Any]]:
        items = clients().core.list_namespaced_service(ns, label_selector=settings.k8s_label_selector, **_TIMEOUT).items
        out = []
        for s in items:
            name = _g(s, "metadata.name")
            try:
                ep = clients().core.read_namespaced_endpoints(name, ns, **_TIMEOUT)
                ready = sum(len(sub.addresses or []) for sub in (ep.subsets or []))
            except Exception:  # noqa: BLE001 - endpoints are supplementary evidence
                ready = None
            out.append(
                {
                    "name": name,
                    "type": _g(s, "spec.type"),
                    "cluster_ip": _g(s, "spec.cluster_ip"),
                    "selector": _g(s, "spec.selector", {}),
                    "ports": [{"port": p.port, "target_port": str(p.target_port)} for p in (_g(s, "spec.ports", []) or [])],
                    "ready_endpoints": ready,
                }
            )
        return out

    def read_logs(ns: str, pod: str, tail: int, previous: bool) -> str:
        text = clients().core.read_namespaced_pod_log(pod, ns, tail_lines=tail, previous=previous, **_TIMEOUT)
        return redact_text(text)[-6000:]

    def list_deployments(a: NsArgs) -> dict[str, Any]:
        ns = ns_of(a.namespace)
        items = clients().apps.list_namespaced_deployment(ns, label_selector=settings.k8s_label_selector, **_TIMEOUT).items
        return {"namespace": ns, "deployments": [deployment_summary(d) for d in items]}

    def list_pods(a: NsArgs) -> dict[str, Any]:
        ns = ns_of(a.namespace)
        return {"namespace": ns, "pods": pods(ns)}

    def get_pod_logs(a: PodLogsArgs) -> dict[str, Any]:
        ns = ns_of(a.namespace)
        return {"namespace": ns, "pod": a.pod_name, "previous": a.previous, "log_tail": read_logs(ns, a.pod_name, a.tail_lines, a.previous)}

    def list_services(a: NsArgs) -> dict[str, Any]:
        ns = ns_of(a.namespace)
        return {"namespace": ns, "services": services(ns)}

    def find_unhealthy(a: NsArgs) -> dict[str, Any]:
        ns = ns_of(a.namespace)
        all_pods = pods(ns)
        return {"namespace": ns, "total": len(all_pods), "unhealthy": [p for p in all_pods if p["problems"]]}

    def deployed_version(a: DeploymentArgs) -> dict[str, Any]:
        ns = ns_of(a.namespace)
        name = a.deployment or settings.k8s_deployment_name
        try:
            dep = clients().apps.read_namespaced_deployment(name, ns, **_TIMEOUT)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError("deployment_not_found", f"Deployment '{name}' not readable in '{ns}': {str(exc)[:150]}") from exc
        summary = deployment_summary(dep)
        cm_data: dict[str, str] = {}
        try:
            cm = clients().core.read_namespaced_config_map(settings.k8s_configmap_name, ns, **_TIMEOUT)
            cm_data = dict(cm.data or {})
        except Exception:  # noqa: BLE001
            cm_data = {}
        return {
            "namespace": ns,
            "deployment": name,
            "image": summary["image"],
            "image_tag": image_tag(summary["image"]),
            "version_label": summary["version_label"],
            "configmap_app_version": cm_data.get("APP_VERSION"),
            "configmap_model_run_id": cm_data.get("MODEL_RUN_ID"),
            "replicas": {"desired": summary["desired"], "ready": summary["ready"]},
        }

    def diagnose(a: DeploymentArgs) -> dict[str, Any]:
        ns = ns_of(a.namespace)
        name = a.deployment or settings.k8s_deployment_name
        core = clients().core
        try:
            dep = deployment_summary(clients().apps.read_namespaced_deployment(name, ns, **_TIMEOUT))
        except Exception as exc:  # noqa: BLE001
            raise ToolError("deployment_not_found", f"Deployment '{name}' not readable in '{ns}': {str(exc)[:150]}") from exc
        findings: list[dict[str, str]] = []
        if (dep["ready"] or 0) < (dep["desired"] or 0):
            findings.append({"problem": "Fewer ready replicas than desired", "evidence": f"ready={dep['ready']}/{dep['desired']}"})
        pod_list = pods(ns)
        events: list[dict[str, Any]] = []
        logs: dict[str, str] = {}
        for p in [x for x in pod_list if x["problems"]]:
            for reason in [*p["waiting_reasons"], *p["last_terminated_reasons"]]:
                if reason in EXPLANATIONS:
                    findings.append({"problem": EXPLANATIONS[reason], "evidence": f"pod {p['name']}: {reason}"})
            if p["phase"] == "Running" and not p["ready"]:
                findings.append({"problem": "Pod is running but failing its readiness probe", "evidence": f"pod {p['name']} not ready"})
            if p["phase"] == "Pending":
                findings.append({"problem": "Pod is Pending - it was never scheduled or its container never started", "evidence": f"pod {p['name']} phase=Pending"})
            try:
                evs = core.list_namespaced_event(ns, field_selector=f"involvedObject.name={p['name']}", **_TIMEOUT).items
                for e in evs[-5:]:
                    if _g(e, "type") == "Warning":
                        events.append({"pod": p["name"], "reason": e.reason, "message": redact_text(str(e.message))[:300]})
            except Exception:  # noqa: BLE001
                pass
            if len(logs) < 2 and p["phase"] in ("Running", "Failed") and (p["restarts"] or p["ready"] is False):
                try:
                    logs[p["name"]] = read_logs(ns, p["name"], 30, previous=bool(p["restarts"]))
                except Exception:  # noqa: BLE001
                    pass
        for ev in events:
            findings.append({"problem": f"Warning event: {ev['reason']}", "evidence": f"pod {ev['pod']}: {ev['message']}"})
        svc = services(ns)
        for s in svc:
            if s["ready_endpoints"] == 0:
                findings.append({"problem": "Service has no ready endpoints (selector mismatch or no ready pods)", "evidence": f"service {s['name']} ready_endpoints=0"})
        return {
            "namespace": ns,
            "deployment": dep,
            "pods": pod_list,
            "services": svc,
            "events": events,
            "log_tail": logs,
            "findings": findings,
            "healthy": not findings,
        }

    def restart(a: DeploymentArgs) -> dict[str, Any]:
        ns = ns_of(a.namespace)  # namespace allow-list is enforced even for confirmed calls
        name = a.deployment or settings.k8s_deployment_name
        stamp = datetime.now(timezone.utc).isoformat()
        body = {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": stamp}}}}}
        clients().apps.patch_namespaced_deployment(name, ns, body, **_TIMEOUT)
        return {"namespace": ns, "deployment": name, "restarted_at": stamp}

    return [
        ToolSpec("k8s_list_deployments", "List application Deployments with desired/ready replicas and image.", NsArgs, list_deployments),
        ToolSpec("k8s_list_pods", "List application Pods with phase, readiness and restarts.", NsArgs, list_pods),
        ToolSpec("k8s_get_pod_logs", "Tail logs of one Pod (optionally the previous container).", PodLogsArgs, get_pod_logs),
        ToolSpec("k8s_list_services", "List Services with ports, selector and ready endpoint count.", NsArgs, list_services),
        ToolSpec("k8s_find_unhealthy_pods", "Find unhealthy Pods and restart events.", NsArgs, find_unhealthy),
        ToolSpec("k8s_get_deployed_version", "Deployed image, version label and ConfigMap version.", DeploymentArgs, deployed_version),
        ToolSpec("k8s_diagnose_deployment", "Evidence-based diagnosis: pods, probes, events, endpoints, logs.", DeploymentArgs, diagnose),
        ToolSpec("k8s_restart_deployment", "Rolling restart of a Deployment (changes state; needs confirmation).", DeploymentArgs, restart, mutating=True),
    ]
