from __future__ import annotations

import pytest

from agent.bootstrap import build_agent
from agent.config import Settings
from tests import fakes

pytest_plugins: list[str] = []


@pytest.fixture
def settings(tmp_path):
    fakes.make_repo(tmp_path)
    return Settings(repo_root=str(tmp_path), airflow_password="pw", tool_timeout_s=3.0,
                    airflow_base_url="http://airflow.test")


@pytest.fixture
def make_agent(settings):
    """Factory returning (agent, handles) wired entirely to fakes."""

    def _make(*, runs=None, session=None, k8s=None, runner=None, mlflow_client=None, cfg=None, llm=None):
        from agent.tools.airflow_tools import AirflowClient

        cfg = cfg or settings
        runs = runs if runs is not None else [fakes.make_run("run-best", 0.93, start=3), fakes.make_run("run-old", 0.88, start=1)]
        mlflow_client = mlflow_client or fakes.FakeMlflowClient(runs)
        session = session or fakes.FakeSession(fakes.airflow_routes())
        factory, clients = k8s or fakes.k8s_factory()
        runner = runner or fakes.make_runner()
        agent = build_agent(
            cfg,
            mlflow_client_factory=lambda: mlflow_client,
            k8s_clients_factory=factory,
            airflow_client=AirflowClient(cfg.airflow_base_url, "airflow", "pw", session=session),
            dvc_runner=runner,
            sleep=lambda s: None,
        )
        if llm is not None:
            agent.llm = llm
        return agent, {"session": session, "k8s": clients, "runner": runner}

    return _make
