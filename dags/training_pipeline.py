"""Airflow 3 DAG: validate -> preprocess -> train -> evaluate/log -> quality gate.

Environment (set in airflow/docker-compose.yaml):
  PROJECT_ROOT          repo mounted into the container (default /opt/airflow/project)
  MLFLOW_TRACKING_URI   MLflow server reachable from the container
  MIN_METRIC_THRESHOLD  quality-gate threshold on f1_macro (default 0.80)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import dag, task

try:
    from airflow.sdk.exceptions import AirflowFailException  # Airflow 3.x
except ImportError:  # pragma: no cover
    from airflow.exceptions import AirflowFailException

from training import train as steps

ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
DATA = ROOT / "data" / "train.csv"
WORK = ROOT / "artifacts" / "airflow"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-agent-classifier")
THRESHOLD = float(os.getenv("MIN_METRIC_THRESHOLD", "0.80"))

DEFAULT_ARGS = {"owner": "mlops", "retries": 2, "retry_delay": timedelta(seconds=30)}


@dag(
    dag_id="training_pipeline",
    schedule=None,  # manual / agent-triggered only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "training"],
    params={"n_estimators": 100, "max_depth": 5, "seed": 42},
)
def training_pipeline():
    @task
    def validate_dataset() -> dict:
        try:
            return steps.validate_dataset(DATA)
        except steps.DatasetValidationError as exc:
            # Deterministic failure: do not burn retries, and block every downstream task.
            raise AirflowFailException(str(exc)) from exc

    @task
    def preprocess_data(validation: dict) -> dict:
        return steps.preprocess(DATA, WORK / "processed")

    @task
    def train_model(split: dict, params=None) -> dict:
        p = (params or {}).get("params", params or {})
        return steps.train_model(
            split["train_path"],
            WORK / "model.pkl",
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=int(p.get("max_depth", 5)),
            seed=int(p.get("seed", 42)),
        ) | {"test_path": split["test_path"]}

    @task
    def evaluate_and_log(trained: dict) -> dict:
        return steps.evaluate_and_log(
            model_path=trained["model_path"],
            test_path=trained["test_path"],
            params={k: trained[k] for k in ("n_estimators", "max_depth", "seed")},
            experiment=EXPERIMENT,
            tracking_uri=TRACKING_URI,
            root=ROOT,
            run_name="airflow-training",
        )

    @task(retries=0)  # a failed gate is a verdict, not a transient fault
    def apply_quality_gate(logged: dict) -> dict:
        try:
            return steps.apply_quality_gate(
                run_id=logged["run_id"],
                metrics=logged["metrics"],
                threshold=THRESHOLD,
                tracking_uri=TRACKING_URI,
                model_uri=logged.get("model_uri"),
            )
        except steps.QualityGateFailed as exc:
            raise AirflowFailException(str(exc)) from exc

    validation = validate_dataset()
    split = preprocess_data(validation)
    trained = train_model(split)
    logged = evaluate_and_log(trained)
    apply_quality_gate(logged)


training_pipeline()
