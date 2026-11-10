"""Central configuration. Every value can be overridden through an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _csv(name: str, default: str) -> frozenset[str]:
    raw = os.getenv(name, default)
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


def _opt_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _opt_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    # --- repository / DVC ---
    repo_root: str = "."
    dvc_file: str = "data/train.csv.dvc"
    # --- MLflow ---
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    default_experiment: str = "mlops-agent-classifier"
    default_metric: str = "f1_macro"
    registered_model_name: str = "mlops-agent-classifier"
    # --- Airflow 3 ---
    airflow_base_url: str = "http://127.0.0.1:8080"
    airflow_username: str = "airflow"
    airflow_password: str = field(default="", repr=False)
    approved_dags: frozenset[str] = frozenset({"training_pipeline"})
    # --- Kubernetes ---
    k8s_allowed_namespaces: frozenset[str] = frozenset({"mlops-agent"})
    k8s_default_namespace: str = "mlops-agent"
    k8s_label_selector: str = "app=mlops-inference"
    k8s_deployment_name: str = "mlops-inference"
    k8s_configmap_name: str = "inference-config"
    # --- retraining policy (evidence rules, never automatic) ---
    min_metric_threshold: float | None = 0.80
    freshness_days: int | None = None
    # --- agent runtime ---
    tool_timeout_s: float = 20.0
    max_agent_steps: int = 12
    llm_provider: str = "rules"  # "rules" (offline, deterministic) or "gemini"
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-2.0-flash"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            repo_root=os.getenv("REPO_ROOT", d.repo_root),
            dvc_file=os.getenv("DVC_FILE", d.dvc_file),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", d.mlflow_tracking_uri),
            default_experiment=os.getenv("MLFLOW_EXPERIMENT_NAME", d.default_experiment),
            default_metric=os.getenv("DEFAULT_METRIC", d.default_metric),
            registered_model_name=os.getenv("REGISTERED_MODEL_NAME", d.registered_model_name),
            airflow_base_url=os.getenv("AIRFLOW_BASE_URL", d.airflow_base_url).rstrip("/"),
            airflow_username=os.getenv("AIRFLOW_USERNAME", d.airflow_username),
            airflow_password=os.getenv("AIRFLOW_PASSWORD", ""),
            approved_dags=_csv("APPROVED_DAGS", "training_pipeline"),
            k8s_allowed_namespaces=_csv("K8S_ALLOWED_NAMESPACES", "mlops-agent"),
            k8s_default_namespace=os.getenv("K8S_NAMESPACE", d.k8s_default_namespace),
            k8s_label_selector=os.getenv("K8S_LABEL_SELECTOR", d.k8s_label_selector),
            k8s_deployment_name=os.getenv("K8S_DEPLOYMENT", d.k8s_deployment_name),
            k8s_configmap_name=os.getenv("K8S_CONFIGMAP", d.k8s_configmap_name),
            min_metric_threshold=_opt_float("MIN_METRIC_THRESHOLD", d.min_metric_threshold),
            freshness_days=_opt_int("FRESHNESS_DAYS", d.freshness_days),
            tool_timeout_s=float(os.getenv("TOOL_TIMEOUT_S", d.tool_timeout_s)),
            max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", d.max_agent_steps)),
            llm_provider=os.getenv("LLM_PROVIDER", d.llm_provider).lower(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", d.gemini_model),
            log_level=os.getenv("LOG_LEVEL", d.log_level),
        )
