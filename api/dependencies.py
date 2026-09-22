"""
api/dependencies.py

FastAPI dependency injection — provides singleton services to route handlers.

Architecture
------------
* AppState holds one SandboxStateStore and a run registry per server process.
* All routes share the same store so evidence always reflects the live state.
* WorkflowRegistry maps run_id → WorkflowRunRecord for retrieval endpoints.
* Dependencies are injected via FastAPI's Depends() system.

WorkflowRunRecord
-----------------
Stores everything needed to answer any GET endpoint for a completed run:
  - CoordinatorResult  — run + records + final status
  - goal               — original NL goal (needed for failure-injection replay)
  - audit_log          — pre-computed AuditEvent list
  - summary            — pre-computed WorkflowSummary
  - injector_config    — failure rules that were active (for provenance)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import Depends

from agents.coordinator import Coordinator, CoordinatorConfig, CoordinatorResult
from agents.executor import Executor
from agents.planner import Planner, PlannerConfig, PlannerMode
from core.audit import AuditEvent, WorkflowSummary, get_audit_log, get_workflow_summary
from sandbox.failure_injection import FailureInjector
from sandbox.state_store import SandboxStateStore
from sandbox.tools import ToolRegistry


# ---------------------------------------------------------------------------
# WorkflowRunRecord — everything needed to serve GET requests for a run
# ---------------------------------------------------------------------------


@dataclass
class WorkflowRunRecord:
    """
    Persisted record for one completed workflow run.

    goal:            Original natural-language goal.
    result:          Full coordinator output (run + action records).
    audit_log:       Pre-built list of AuditEvent objects.
    summary:         Pre-built WorkflowSummary with aggregate metrics.
    injector_config: Serialized failure rules that were active (for provenance).
    created_at:      When the run was started.
    parent_run_id:   If this was a failure-injection replay, the original run_id.
    """

    goal: str
    result: CoordinatorResult
    audit_log: list[AuditEvent] = field(default_factory=list)
    summary: WorkflowSummary | None = None
    injector_config: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    parent_run_id: str | None = None

    @property
    def run_id(self) -> str:
        return self.result.run.run_id


# ---------------------------------------------------------------------------
# Singleton application state
# ---------------------------------------------------------------------------


class AppState:
    """
    Single instance shared across all requests.

    In production this would be backed by Redis/Postgres; here it is
    in-memory and reset on server restart.

    Stores two parallel registries for backward compatibility:
      _registry         → CoordinatorResult (used by legacy /workflows/ routes)
      _workflow_records → WorkflowRunRecord  (used by new /workflow/ routes)
    """

    def __init__(self) -> None:
        self.store = SandboxStateStore()
        self._registry: dict[str, CoordinatorResult] = {}
        self._workflow_records: dict[str, WorkflowRunRecord] = {}

    # ------------------------------------------------------------------
    # Legacy registry (used by /workflows/ routes)
    # ------------------------------------------------------------------

    def register_result(self, result: CoordinatorResult) -> None:
        self._registry[result.run.run_id] = result

    def get_result(self, run_id: str) -> CoordinatorResult | None:
        return self._registry.get(run_id)

    def list_results(self) -> list[CoordinatorResult]:
        return list(self._registry.values())

    # ------------------------------------------------------------------
    # Workflow record registry (used by /workflow/ routes)
    # ------------------------------------------------------------------

    def register_record(self, record: WorkflowRunRecord) -> None:
        self._workflow_records[record.run_id] = record
        # Also register in legacy registry for cross-route compatibility
        self._registry[record.run_id] = record.result

    def get_record(self, run_id: str) -> WorkflowRunRecord | None:
        return self._workflow_records.get(run_id)

    def list_records(self) -> list[WorkflowRunRecord]:
        return list(self._workflow_records.values())

    # ------------------------------------------------------------------
    # Reset (used by tests)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Wipe all sandbox state and run history."""
        self.store = SandboxStateStore()
        self._registry.clear()
        self._workflow_records.clear()


# Module-level singleton — created once at import time
_app_state = AppState()


# ---------------------------------------------------------------------------
# FastAPI dependency providers
# ---------------------------------------------------------------------------


def get_app_state() -> AppState:
    return _app_state


def get_store(state: Annotated[AppState, Depends(get_app_state)]) -> SandboxStateStore:
    return state.store


def get_planner(mode: str = "mock") -> Planner:
    planner_mode = PlannerMode.LLM if mode == "llm" else PlannerMode.MOCK
    return Planner(PlannerConfig(mode=planner_mode))


def make_coordinator(
    store: SandboxStateStore,
    max_retries: int = 3,
    max_replan_cycles: int = 2,
    atomic_completion: bool = False,
    injector: FailureInjector | None = None,
) -> Coordinator:
    """Build a Coordinator wired to the given store and optional injector."""
    registry = ToolRegistry(store, injector or FailureInjector())
    executor = Executor(registry)
    config = CoordinatorConfig(
        max_retries=max_retries,
        max_replan_cycles=max_replan_cycles,
        atomic_completion=atomic_completion,
    )
    return Coordinator(store, executor, config)


def build_and_run(
    goal: str,
    store: SandboxStateStore | None = None,
    max_retries: int = 3,
    max_replan_cycles: int = 2,
    atomic_completion: bool = False,
    injector: FailureInjector | None = None,
    planner_mode: str = "mock",
    parent_run_id: str | None = None,
    injector_config: list[dict[str, Any]] | None = None,
) -> WorkflowRunRecord:
    """
    Plan and execute a goal, return a fully-populated WorkflowRunRecord.

    This is the canonical helper used by both /workflow/run and
    /workflow/{id}/failure-injection.  It:
      1. Creates a fresh SandboxStateStore (or uses the provided one)
      2. Runs Planner → Coordinator
      3. Computes audit log + summary
      4. Returns a WorkflowRunRecord ready to be stored

    Raises
    ------
    ValueError  — if the goal is empty or the planner rejects it.
    RuntimeError — if an unexpected internal error occurs (should not happen
                   in normal operation; indicates a programming error).
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    goal = goal.strip()
    if not goal:
        raise ValueError("goal must not be empty")

    try:
        use_store = store or SandboxStateStore()
        coordinator = make_coordinator(
            use_store,
            max_retries=max_retries,
            max_replan_cycles=max_replan_cycles,
            atomic_completion=atomic_completion,
            injector=injector,
        )
        planner = get_planner(planner_mode)

        plan = planner.plan(goal)
        if not plan.steps:
            raise ValueError(
                f"Planner produced an empty plan for goal: {goal!r}. "
                "Try a different goal or check the mock planner's keyword coverage."
            )

        result = coordinator.execute_plan(plan)

        audit = get_audit_log(result)
        summary = get_workflow_summary(result)

        return WorkflowRunRecord(
            goal=goal,
            result=result,
            audit_log=audit,
            summary=summary,
            injector_config=injector_config or [],
            parent_run_id=parent_run_id,
        )
    except (ValueError, TypeError) as exc:
        _log.warning("build_and_run validation error for goal %r: %s", goal, exc)
        raise
    except Exception as exc:
        _log.exception("build_and_run unexpected error for goal %r", goal)
        raise RuntimeError(
            f"Internal error executing workflow for goal {goal!r}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Environment helpers (LLM config — never exposed in API responses)
# ---------------------------------------------------------------------------


def get_llm_config() -> dict[str, str | None]:
    """
    Read LLM configuration from environment variables.
    Never returned in API responses — for internal use only.
    """
    return {
        "provider": os.getenv("LLM_PROVIDER"),               # openai | anthropic | google
        "model": os.getenv("LLM_MODEL"),                      # e.g. gpt-4o
        # Keys intentionally NOT included here — use env vars directly
        # OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
    }


def llm_is_configured() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
