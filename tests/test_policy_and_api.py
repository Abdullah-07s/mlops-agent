from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent import policy
from api.main import create_app


@pytest.mark.parametrize("msg", [
    "Trigger the model training pipeline", "please run the training pipeline",
    "check the status and then trigger the training dag", "Can you start the training workflow?",
])
def test_explicit_trigger_detected(msg):
    assert policy.explicit_trigger(msg)


@pytest.mark.parametrize("msg", [
    "Find the best recent model ... and explain whether retraining is necessary.",
    "Show me the last run of the training pipeline", "Did the training pipeline succeed?",
])
def test_non_requests_do_not_trigger(msg):
    assert not policy.explicit_trigger(msg)


def test_restart_detection():
    assert policy.explicit_restart("please restart the deployment")
    assert not policy.explicit_restart("why does the pod keep restarting")


def test_api_chat_roundtrip(make_agent):
    agent, _ = make_agent()
    client = TestClient(create_app(agent=agent))
    r = client.post("/chat", json={"message": "check the deployment"}, headers={"x-request-id": "abc123"})
    body = r.json()
    assert r.status_code == 200 and r.headers["x-request-id"] == "abc123" and body["request_id"] == "abc123"
    assert body["facts"] and body["tool_trace"][0]["status"] == "ok"


def test_api_validates_input(make_agent):
    agent, _ = make_agent()
    client = TestClient(create_app(agent=agent))
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 3000}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_api_health_ready_and_tools(make_agent):
    agent, _ = make_agent()
    client = TestClient(create_app(agent=agent))
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/ready").json()["tools"] == 26
    assert any(t["name"] == "k8s_restart_deployment" and t["mutating"] for t in client.get("/tools").json())
