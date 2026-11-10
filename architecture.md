# Architecture

## Layers and data flow
```
api/main.py            HTTP: validation, request-id, JSON contract (no business logic)
agent/orchestrator.py  loop: plan -> policy -> execute -> repeat -> correlate -> answer
agent/llm.py           planners: RuleBasedLLM (deterministic) | GeminiLLM (REST) + fallback
agent/policy.py        "did the USER ask for this change?" (independent of the planner)
agent/registry.py      validate args -> authorize -> run with timeout -> classify -> log
agent/tools/*          one module per system; each exposes build_tools(settings, <client>)
agent/correlator.py    results -> facts / uncertainties / recommendations (deterministic)
agent/report.py        rendering + grounding check for LLM narratives
agent/bootstrap.py     composition root (the only place concrete clients are wired)
training/, dags/       the pipeline the agent observes; training/provenance.py is shared
inference/             the FastAPI model service deployed to Kubernetes
```
Request flow: `POST /chat` -> `Agent.handle` -> planner picks next tool -> policy check ->
`ToolRegistry.execute` -> result appended to history -> repeat (max 12 steps, no repeated call)
-> `correlator` builds facts/uncertainties/recommendations -> writer phrases them -> response with
the complete `tool_trace`.

Provenance chain: Airflow/`train.py` tags each MLflow run with `dvc_md5` + `git_commit`
-> agent reads the tags -> `dvc_find_commit_for_hash` (`git log -S<md5>`) -> exact dataset commit
-> `dvc_reproduce_steps` explains how to restore it.

## Tools and permissions
| Module | Tools | Access |
|---|---|---|
| DVC (6) | list_datasets, current_version, compare_versions, detect_changes, find_commit_for_hash, reproduce_steps | read-only; allow-list `git rev-parse/log/show/status`, `dvc status/diff`; revisions/paths regex-validated; no shell |
| MLflow (6) | list_experiments, list_runs, get_run, best_run, compare_runs, get_registry_info | read-only via `MlflowClient` |
| Airflow (6) | list_dags, get_dag_runs, get_task_instances, get_failed_tasks, get_task_logs, **trigger_training_dag** | JWT bearer; trigger only for `APPROVED_DAGS`, needs user intent + confirmation |
| Kubernetes (8) | list_deployments, list_pods, get_pod_logs, list_services, find_unhealthy_pods, get_deployed_version, diagnose_deployment, **restart_deployment** | every call restricted to approved namespaces; restart needs user intent + confirmation |

Safety invariants (each has a test): unknown tool/invalid args never execute; mutating tools
need explicit request text (policy) AND `confirm_mutations` (registry); namespace allow-list is
enforced even for confirmed calls; the agent never starts retraining from a status question;
logs/errors are redacted; an LLM narrative containing numbers absent from the evidence is
discarded and replaced by the deterministic report.

## Design review (senior-engineer pass over this codebase)
**Decisions that avoid common failures**
- Safety lives in registry + policy, not in prompts, so changing the LLM cannot weaken it.
- Planner and writer are behind one small protocol; the offline planner doubles as the fallback
  and the test double, so the system degrades instead of failing when the LLM is down.
- Facts are derived from tool output by code; the LLM only phrases them.
- Provenance helpers read `.git`/`.dvc` files directly, so they work in the Airflow container.

**Critical problem areas (known limitations)**
1. Registry thread-pool timeouts cannot kill a hung call; the thread lingers until the client
   library's own timeout fires (mitigated: short MLflow/Kubernetes/requests timeouts).
2. The rule-based planner is regex intent matching: transparent and testable, but it will miss
   unusual phrasings (Gemini mode covers them).
3. `git log -S` finds the *introducing* commit only if the `.dvc` file was committed; a run
   trained on uncommitted data is reported as "hash not in git history".
4. Tool results are held in memory per request; no cross-request cache, so repeated questions
   re-query every system.
5. Airflow demo uses `_PIP_ADDITIONAL_REQUIREMENTS`; production would bake a custom image.

**Refactoring strategies if this grows**
- Replace the `_plan` regex table with declarative playbooks (intent -> ordered steps with
  dependencies) loaded from YAML; add a per-tool TTL cache in the registry.
- Move `correlator.SUMMARIZERS` next to each tool module (tool + summarizer + tests together).
- Run tools in a process pool (or async clients) for real cancellation and parallel fan-out.
- Add authN/Z on the agent API and a Kubernetes Role scoped to the agent's namespace.
