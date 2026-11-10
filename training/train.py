"""Training pipeline stages. The Airflow DAG calls these one task at a time;
`python -m training.train` (and the dvc.yaml stage) runs them end to end."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from training.provenance import dataset_provenance

FEATURES = [f"feature_{i}" for i in range(5)]
TARGET = "target"


class DatasetValidationError(ValueError):
    """The dataset is unusable; retrying cannot help."""


class QualityGateFailed(RuntimeError):
    """The candidate model did not meet the acceptance threshold."""


def validate_dataset(csv_path: str | Path, *, min_rows: int = 100) -> dict[str, Any]:
    path = Path(csv_path)
    if not path.exists():
        raise DatasetValidationError(f"dataset not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"missing columns: {missing}")
    if len(df) < min_rows:
        raise DatasetValidationError(f"only {len(df)} rows; need at least {min_rows}")
    if df[[*FEATURES, TARGET]].isnull().any().any():
        raise DatasetValidationError("dataset contains null values")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise DatasetValidationError("target must be binary (0/1)")
    if df[TARGET].nunique() < 2:
        raise DatasetValidationError("target contains a single class")
    return {"rows": int(len(df)), "positive_rate": float(df[TARGET].mean())}


def preprocess(csv_path: str | Path, out_dir: str | Path, seed: int = 42) -> dict[str, Any]:
    df = pd.read_csv(csv_path).drop_duplicates()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[TARGET])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path, test_path = out / "train_split.csv", out / "test_split.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return {"train_path": str(train_path), "test_path": str(test_path), "train_rows": len(train_df), "test_rows": len(test_df)}


def train_model(
    train_path: str | Path, model_path: str | Path, *, n_estimators: int = 100, max_depth: int = 5, seed: int = 42
) -> dict[str, Any]:
    df = pd.read_csv(train_path)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=1)
    model.fit(df[FEATURES], df[TARGET])
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {"model_path": str(model_path), "n_estimators": n_estimators, "max_depth": max_depth, "seed": seed}


def _log_model(model: Any) -> str | None:
    import mlflow.sklearn

    # cloudpickle: MLflow >= 3.? defaults to skops, which rejects tree ensembles as "untrusted".
    # The model is produced by this pipeline itself, so pickle-based loading is acceptable here.
    fmt = {"serialization_format": "cloudpickle"}
    try:
        info = mlflow.sklearn.log_model(model, name="model", **fmt)  # MLflow 3
    except TypeError:
        info = mlflow.sklearn.log_model(model, artifact_path="model", **fmt)  # MLflow 2
    return getattr(info, "model_uri", None)


def evaluate_and_log(
    *,
    model_path: str | Path,
    test_path: str | Path,
    params: dict[str, Any],
    experiment: str,
    tracking_uri: str,
    root: str | Path = ".",
    dvc_file: str = "data/train.csv.dvc",
    run_name: str | None = None,
) -> dict[str, Any]:
    import mlflow

    model = joblib.load(model_path)
    test = pd.read_csv(test_path)
    preds = model.predict(test[FEATURES])
    metrics = {
        "accuracy": float(accuracy_score(test[TARGET], preds)),
        "f1_macro": float(f1_score(test[TARGET], preds, average="macro")),
    }
    prov = dataset_provenance(root, dvc_file)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "dvc_md5": prov["dvc_md5"] or "unknown",
            "git_commit": prov["git_commit"] or "unknown",
            "dvc_file": prov["dvc_file"],
            "quality_gate_passed": "pending",
        })
        mlflow.log_artifact(str(model_path), artifact_path="model_file")
        model_uri = _log_model(model)
        run_id = run.info.run_id
    return {"run_id": run_id, "metrics": metrics, "model_uri": model_uri}


def _promote_candidate(tracking_uri: str, model_uri: str | None, model_name: str) -> str | None:
    """Best-effort registry promotion; failure must never mask a passed gate."""
    if not model_uri:
        return None
    try:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        version = mlflow.register_model(model_uri, model_name)
        MlflowClient(tracking_uri=tracking_uri).set_registered_model_alias(model_name, "candidate", version.version)
        return str(version.version)
    except Exception as exc:  # noqa: BLE001
        print(f"registry promotion skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def apply_quality_gate(
    *,
    run_id: str,
    metrics: dict[str, float],
    threshold: float,
    tracking_uri: str,
    model_uri: str | None = None,
    model_name: str = "mlops-agent-classifier",
    metric_name: str = "f1_macro",
) -> dict[str, Any]:
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    value = metrics[metric_name]
    if value < threshold:
        client.set_tag(run_id, "quality_gate_passed", "false")
        raise QualityGateFailed(f"{metric_name}={value:.4f} is below threshold {threshold:.2f}")
    client.set_tag(run_id, "quality_gate_passed", "true")
    version = _promote_candidate(tracking_uri, model_uri, model_name)
    return {"passed": True, metric_name: value, "registered_version": version}


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    validate_dataset(root / args.data)
    split = preprocess(root / args.data, root / "artifacts" / "processed", args.seed)
    model_path = root / "models" / "model.pkl"
    train_model(split["train_path"], model_path, n_estimators=args.n_estimators, max_depth=args.max_depth, seed=args.seed)
    logged = evaluate_and_log(
        model_path=model_path,
        test_path=split["test_path"],
        params={"n_estimators": args.n_estimators, "max_depth": args.max_depth, "seed": args.seed},
        experiment=args.experiment,
        tracking_uri=args.tracking_uri,
        root=root,
        run_name=args.run_name,
    )
    (root / "metrics.json").write_text(json.dumps(logged["metrics"], indent=2))
    gate = apply_quality_gate(
        run_id=logged["run_id"], metrics=logged["metrics"], threshold=args.threshold,
        tracking_uri=args.tracking_uri, model_uri=logged["model_uri"],
    )
    return {**logged, "gate": gate}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate -> preprocess -> train -> evaluate/log -> quality gate")
    p.add_argument("--root", default=".")
    p.add_argument("--data", default="data/train.csv")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=float(os.getenv("MIN_METRIC_THRESHOLD", "0.80")))
    p.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-agent-classifier"))
    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    p.add_argument("--run-name", default=None)
    args = p.parse_args(argv)
    try:
        result = run_pipeline(args)
    except (DatasetValidationError, QualityGateFailed) as exc:
        print(f"PIPELINE STOPPED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": result["run_id"], "metrics": result["metrics"], "gate": result["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
