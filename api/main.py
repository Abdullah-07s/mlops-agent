"""FastAPI interface for the agent: validation in, evidence-based response out."""

from __future__ import annotations

import threading

from fastapi import FastAPI, Request, Response

from agent.bootstrap import build_agent
from agent.config import Settings
from agent.logging_utils import configure_logging, new_request_id
from agent.orchestrator import Agent
from agent.schemas import AgentResponse, ChatRequest


def create_app(agent: Agent | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="MLOps Agent API", version="1.0.0")
    holder: dict[str, Agent] = {}
    lock = threading.Lock()

    def get_agent() -> Agent:
        with lock:  # build lazily so importing the app never touches external services
            if "agent" not in holder:
                holder["agent"] = agent or build_agent(settings)
            return holder["agent"]

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("x-request-id") or new_request_id()
        request.state.request_id = rid
        response: Response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    @app.post("/chat", response_model=AgentResponse)
    def chat(body: ChatRequest, request: Request) -> AgentResponse:
        return get_agent().handle(body, request_id=request.state.request_id)

    @app.get("/tools")
    def tools() -> list[dict]:
        return get_agent().registry.catalog()

    @app.get("/health")
    def health() -> dict:
        return {"status": "healthy"}

    @app.get("/ready")
    def ready() -> dict:
        return {"ready": True, "tools": len(get_agent().registry.names()), "llm": get_agent().llm.name}

    return app


configure_logging(Settings.from_env().log_level)
app = create_app()
