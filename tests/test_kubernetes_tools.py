from __future__ import annotations

from types import SimpleNamespace as NS

from agent.registry import ExecutionContext, ToolRegistry
from agent.schemas import ToolCall, ToolStatus
from agent.tools import kubernetes_tools
from tests import fakes
from tests.fakes import make_deployment, make_pod

READ = ExecutionContext("t")
CONFIRMED = ExecutionContext("t", confirmed=True)


def make(settings, **kw):
    factory, clients = fakes.k8s_factory(**kw)
    r = ToolRegistry()
    r.register_all(kubernetes_tools.build_tools(settings, clients_factory=factory))
    return r, clients


def run(r, tool, ctx=READ, **args):
    return r.execute(ToolCall(tool=tool, args=args), ctx)


def test_list_pods_and_deployments(settings):
    r, _ = make(settings)
    assert len(run(r, "k8s_list_pods").data["pods"]) == 2
    assert run(r, "k8s_list_deployments").data["deployments"][0]["desired"] == 2


def test_unhealthy_pod_detection(settings):
    pods = [make_pod("ok"), make_pod("bad", ready=False, restarts=4, waiting="CrashLoopBackOff", terminated="OOMKilled")]
    r, _ = make(settings, pods=pods)
    bad = run(r, "k8s_find_unhealthy_pods").data["unhealthy"]
    assert [p["name"] for p in bad] == ["bad"]
    assert "waiting:CrashLoopBackOff" in bad[0]["problems"] and "restarts=4" in bad[0]["problems"]


def test_deployed_version_reports_all_sources(settings):
    r, _ = make(settings, app_version="0.2.0")
    d = run(r, "k8s_get_deployed_version").data
    assert d["image_tag"] == "0.1.0" and d["configmap_app_version"] == "0.2.0"


def test_diagnose_image_pull_failure(settings):
    pods = [make_pod("p", phase="Pending", ready=False, waiting="ImagePullBackOff")]
    ev = NS(type="Warning", reason="Failed", message="Failed to pull image: not found")
    r, _ = make(settings, pods=pods, deployment=make_deployment(ready=0), events=[ev])
    d = run(r, "k8s_diagnose_deployment").data
    problems = " | ".join(f["problem"] for f in d["findings"])
    assert not d["healthy"] and "image cannot be pulled" in problems and "Fewer ready replicas" in problems


def test_diagnose_flags_service_without_endpoints(settings):
    r, _ = make(settings, endpoints=0)
    assert any("no ready endpoints" in f["problem"] for f in run(r, "k8s_diagnose_deployment").data["findings"])


def test_logs_are_redacted(settings):
    r, _ = make(settings)
    out = run(r, "k8s_get_pod_logs", pod_name="p1").data["log_tail"]
    assert "abc123secret" not in out


def test_unauthorized_namespace_is_rejected(settings):
    r, _ = make(settings)
    res = run(r, "k8s_list_pods", namespace="kube-system")
    assert res.status == ToolStatus.REJECTED and res.error_code == "namespace_not_allowed"


def test_restart_without_confirmation_is_rejected(settings):
    r, clients = make(settings)
    res = run(r, "k8s_restart_deployment")
    assert res.error_code == "confirmation_required" and clients.apps.patches == []


def test_confirmed_restart_in_forbidden_namespace_is_still_rejected(settings):
    r, clients = make(settings)
    res = run(r, "k8s_restart_deployment", CONFIRMED, namespace="kube-system")
    assert res.error_code == "namespace_not_allowed" and clients.apps.patches == []


def test_confirmed_restart_in_approved_namespace_patches(settings):
    r, clients = make(settings)
    assert run(r, "k8s_restart_deployment", CONFIRMED).ok
    name, ns, body = clients.apps.patches[0]
    assert ns == "mlops-agent" and "restartedAt" in str(body)
