"""
scenarios/scenario3_unrecoverable.py

SCENARIO 3 — UNRECOVERABLE FAILURE / BLOCKED

The create_project tool is injected with a permanent EXECUTION_FAILURE
(fail_count=999 — effectively unlimited).  Every attempt fails.

Recovery loop:
  Attempt 1 → FAIL → RecoveryAgent: RETRY (2 retries left)
  Attempt 2 → FAIL → RecoveryAgent: RETRY (1 retry left)
  Attempt 3 → FAIL → RecoveryAgent: ABORT (0 retries left)
  Coordinator: BLOCKED

Final status: BLOCKED

The system explicitly records WHY it refused to claim completion:
  - blocked_reason in WorkflowSummary
  - BLOCKED in transition_history with reason string
  - No SUCCEEDED actions in the run
"""

from __future__ import annotations

from agents.coordinator import CoordinatorConfig
from sandbox.failure_injection import FailureInjector, FailureMode, FailureRule
from scenarios.runner import ScenarioResult, ScenarioRunner

GOAL = (
    "Create project Hackathon Alpha, add Harshit as a member, "
    "make the project private, and generate a report."
)

SCENARIO_NAME = "Scenario 3 — Unrecoverable Failure (BLOCKED)"

_PROJECT_ID = "hackathon-alpha"
MAX_RETRIES = 2  # bounded retry limit for demo clarity


def _build_injector() -> FailureInjector:
    """Permanent EXECUTION_FAILURE on create_project — never recovers."""
    injector = FailureInjector()
    injector.register(FailureRule(
        tool_name="create_project",
        mode=FailureMode.EXECUTION_FAILURE,
        reason=(
            "PERMANENT FAILURE: database write rejected. "
            "Project quota exceeded — contact administrator."
        ),
        param_filter={"project_id": _PROJECT_ID},
        fail_count=999,   # effectively permanent
    ))
    return injector


def run() -> ScenarioResult:
    """
    Execute the unrecoverable-failure scenario deterministically.

    Returns a ScenarioResult with:
      - final_status == BLOCKED
      - summary.blocked_reason explains why completion was refused
      - retry_count == MAX_RETRIES (bounded, never infinite)
      - no SUCCEEDED actions
      - sandbox is clean (project never written)
    """
    injector = _build_injector()
    runner = ScenarioRunner(
        injector=injector,
        config=CoordinatorConfig(
            max_retries=MAX_RETRIES,
            max_replan_cycles=0,  # no replanning — demonstrate pure retry→block path
            atomic_completion=False,
        ),
    )
    result = runner.run(
        goal=GOAL,
        scenario_name=SCENARIO_NAME,
        expected_status="BLOCKED",
        extra={
            "max_retries": MAX_RETRIES,
            "failure_tool": "create_project",
        },
    )
    return result
