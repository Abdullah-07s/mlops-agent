# Assignment 3 - Production-Ready MLOps AI Agent

A conversational agent that answers MLOps questions **from tool evidence**, using DVC (data
provenance), MLflow (experiments/registry), Airflow 3 (training DAG) and Kubernetes (deployment).

```
User -> FastAPI (api/) -> Agent (agent/orchestrator.py)
                           |-- planner: rules (offline) or Gemini  --+
                           |-- policy: mutations only if user asked  |  every call goes through
                           '-- ToolRegistry: validate/authorize/timeout/log
                                |-- DVC  |-- MLflow  |-- Airflow  |-- Kubernetes
        -> facts + uncertainties + recommended actions (+ full tool trace)
```
See `architecture.md` for the design, permission table and review notes.

## Install
```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements-ml.txt
pip install -e ".[dev,mlflow,k8s]"
pip install dvc
```

## Configuration (environment variables)
| Variable | Default | Purpose |
|---|---|---|
| `REPO_ROOT` | `.` | repo containing `.git` and `*.dvc` |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | MLflow server |
| `MLFLOW_EXPERIMENT_NAME` | `mlops-agent-classifier` | default experiment |
| `AIRFLOW_BASE_URL` / `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD` | `http://127.0.0.1:8080` / `airflow` / *(empty)* | Airflow 3 API (JWT) |
| `APPROVED_DAGS` | `training_pipeline` | DAGs the agent may trigger |
| `K8S_NAMESPACE` / `K8S_ALLOWED_NAMESPACES` | `mlops-agent` | the only namespace(s) the agent may touch |
| `MIN_METRIC_THRESHOLD` | `0.80` | retraining policy: metric floor (f1_macro) |
| `FRESHNESS_DAYS` | unset | retraining policy: max model age |
| `LLM_PROVIDER` | `rules` | `rules` (offline) or `gemini` |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | - / `gemini-2.0-flash` | only for `gemini` |

Never commit keys, kubeconfigs or tokens; use environment variables.

## Launch each service
1. **MLflow**: `mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 0.0.0.0 --port 5000`
2. **Training / DVC**: `python -m training.make_dataset --rows 500`, `dvc add data/train.csv`, `dvc repro`
3. **Kubernetes**: `docker build -t mlops-inference:0.1.0 .` then `kubectl apply -f kubernetes/`
4. **Airflow 3**: official compose + `airflow/docker-compose.override.yaml` (see GUIDE.md step 5)
5. **Agent API**: `uvicorn api.main:app --port 8000` (docs at `/docs`)

## Use the agent
```powershell
python -m agent.main "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary." --evidence
python -m agent.main "Why is my prediction API unavailable?"
python -m agent.main "Trigger the model training pipeline and report whether execution succeeded." --confirm
Invoke-RestMethod -Method Post http://127.0.0.1:8000/chat -ContentType application/json -Body '{"message":"check the deployment"}'
```
State-changing tools (Airflow trigger, Deployment restart) run only if the message explicitly
asks for them **and** `--confirm` / `"confirm_mutations": true` is supplied.

## Tests
```powershell
ruff check .
pytest
```
All external services are mocked; no cluster, Airflow, MLflow server or API key is needed.

Data note: the DVC remote used in the lab is a local folder (../dvc-storage). Regenerate data with python -m training.make_dataset --rows 500 and run dvc repro.
