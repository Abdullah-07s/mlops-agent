"""Inference service deployed to Kubernetes (the workload the agent inspects)."""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("inference")


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=1, max_length=100)


class ModelHolder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._model = None

    def get(self):
        if self._model is None:
            if not self.path.exists():
                raise FileNotFoundError(f"model file not found: {self.path}")
            self._model = joblib.load(self.path)
            log.info("model loaded from %s", self.path)
        return self._model


def create_app(model_path: str | None = None) -> FastAPI:
    app = FastAPI(title="MLOps Inference Service", version=os.getenv("APP_VERSION", "dev"))
    holder = ModelHolder(Path(model_path or os.getenv("MODEL_PATH", "models/model.pkl")))
    meta = lambda: {  # noqa: E731
        "version": os.getenv("APP_VERSION", "dev"),
        "model_run_id": os.getenv("MODEL_RUN_ID", "unset"),
        "hostname": socket.gethostname(),
    }

    @app.get("/health")
    def health() -> dict:
        return {"status": "healthy", "hostname": socket.gethostname()}

    @app.get("/ready")
    def ready():
        try:
            holder.get()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=503, content={"ready": False, "reason": str(exc)[:200]})
        return {"ready": True, **meta()}

    @app.get("/version")
    def version() -> dict:
        return meta()

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict:
        try:
            model = holder.get()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail="model not loaded") from exc
        if len(req.features) != model.n_features_in_:
            raise HTTPException(status_code=422, detail=f"expected {model.n_features_in_} features")
        return {"prediction": int(model.predict([req.features])[0]), **meta()}

    return app


app = create_app()
