"""
scenarios/runner.py

ScenarioRunner — canonical entry point for all four demo scenarios.

Wraps Planner + Coordinator with all necessary wiring and returns a
ScenarioResult that includes the CoordinatorResult, audit log, and
workflow summary.  Every scenario is fully deterministic and requires
no API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.coordinator import Coordinator, CoordinatorConfig, CoordinatorResult
from agents.executor import Executor
from agents.planner import Planner, PlannerConfig, PlannerMode
from core.audit import AuditEvent, WorkflowSummary, get_audit_log, get_workflow_summary
from core.models import WorkflowStatus
from sandbox.failure_injection import FailureInjector
from sandbox.state_store import SandboxStateStore
from sandbox.tools import ToolRegistry


# ---------------------------------------------------------------------------
# ScenarioResult
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """
    Output of running one hackathon demo scenario.

    Attributes
    ----------
    scenario_name:          Human-readable scenario identifier.
    goal:                   The natural-language goal that was planned.
    coordinator_result:     Full CoordinatorResult (run + records + summary).
    audit_log:              Ordered list of AuditEvents for the full run.
    summary:                WorkflowSummary with aggregate metrics.
    rollback_verifications: Per-entity sandbox checks after rollback (Scenario 4).
    extra:                  Any extra scenario-specific data.
    """

    scenario_name: str
    goal: str
    coordinator_result: CoordinatorResult
    audit_log: list[AuditEvent]
    summary: WorkflowSummary
    rollback_verifications: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def final_status(self) -> WorkflowStatus:
        return self.coordinator_result.run.status

    @property
    def passed(self) -> bool:
        """True if the scenario produced the expected terminal state."""
        expected = self.extra.get("expected_status")
        if expected is None:
            return True
        return self.final_status.value == expected


# ---------------------------------------------------------------------------
# ScenarioRunner
# ---------------------------------------------------------------------------


class ScenarioRunner:
    """
    Builds a fully-wired coordinator from a given store + injector and
    runs a goal string through the Planner → Coordinator pipeline.

    Parameters
    ----------
    store:    SandboxStateStore to use (fresh one created if None).
    injector: FailureInjector (empty/no-op if None).
    config:   CoordinatorConfig (defaults used if None).
    """

    def __init__(
        self,
        store: SandboxStateStore | None = None,
        injector: FailureInjector | None = None,
        config: CoordinatorConfig | None = None,
    ) -> None:
        self._store = store or SandboxStateStore()
        self._injector = injector or FailureInjector()
        self._config = config or CoordinatorConfig()

        registry = ToolRegistry(self._store, self._injector)
        executor = Executor(registry)
        self._coordinator = Coordinator(self._store, executor, self._config)
        self._planner = Planner(PlannerConfig(mode=PlannerMode.MOCK))

    @property
    def store(self) -> SandboxStateStore:
        return self._store

    def run(
        self,
        goal: str,
        scenario_name: str = "unnamed",
        expected_status: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ScenarioResult:
        """
        Plan and execute *goal*, return a fully-populated ScenarioResult.

        Steps
        -----
        1. Planner converts goal → TaskPlan
        2. Coordinator executes with evidence gating
        3. Audit log and summary are computed post-run
        """
        plan = self._planner.plan(goal)
        result = self._coordinator.execute_plan(plan)

        audit = get_audit_log(result)
        summary = get_workflow_summary(result)

        return ScenarioResult(
            scenario_name=scenario_name,
            goal=goal,
            coordinator_result=result,
            audit_log=audit,
            summary=summary,
            extra={"expected_status": expected_status, **(extra or {})},
        )
