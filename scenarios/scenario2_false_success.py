"""
scenarios/scenario2_false_success.py

SCENARIO 2 — FALSE SUCCESS DETECTION AND RECOVERY

The add_member tool is injected to return a successful-looking execution
response on the FIRST call while NOT actually mutating sandbox state
(FALSE_SUCCESS mode, max_fires=1).

Evidence flow on attempt 1:
  Executor:  "Member added successfully."  ← self-reported claim
  Sandbox:   member_exists = False         ← machine-observed truth
  Verifier:  FAIL — executor claim contradicts machine-observed state

Recovery:
  RecoveryAgent classifies → STATE_MISMATCH → RETRY
  Second attempt: FALSE_SUCCESS rule exhausted → real tool executes
  Evidence: member_exists = True → PASS

Final: VERIFIED COMPLETED
"""

from __future__ import annotations

from agents.coordinator import CoordinatorConfig
from core.models import WorkflowStatus
from sandbox.failure_injection import FailureInjector, FailureMode, FailureRule
from scenarios.runner import ScenarioResult, ScenarioRunner

GOAL = (
    "Create project Hackathon Alpha, add Harshit as a member, "
    "make the project private, and generate a report."
)

SCENARIO_NAME = "Scenario 2 — False Success Detection"

# Project ID as extracted by the MockPlanner from the goal above
_PROJECT_ID = "hackathon-alpha"


def _build_injector() -> FailureInjector:
    """
    Inject FALSE_SUCCESS on add_member exactly once.

    max_fires=1 ensures the rule fires only on the first call.
    The second call goes through to the real tool and mutates state.
    """
    injector = FailureInjector()
    injector.register(FailureRule(
        tool_name="add_member",
        mode=FailureMode.FALSE_SUCCESS,
        reason=(
            "INJECTED: executor claim contradicts machine-observed state. "
            "Member record was NOT written to sandbox."
        ),
        param_filter={"project_id": _PROJECT_ID},
        max_fires=1,  # fires ONCE then becomes a no-op
    ))
    return injector


def run() -> ScenarioResult:
    """
    Execute the false-success scenario deterministically.

    Returns a ScenarioResult with:
      - final_status == COMPLETED (second attempt succeeds)
      - at least one FAIL verdict in audit log (caught the lie)
      - at least one RETRY recovery decision
      - add_member action has retry_count >= 1
    """
    injector = _build_injector()
    runner = ScenarioRunner(
        injector=injector,
        config=CoordinatorConfig(
            max_retries=3,
            max_replan_cycles=2,
            atomic_completion=False,
        ),
    )
    result = runner.run(
        goal=GOAL,
        scenario_name=SCENARIO_NAME,
        expected_status="COMPLETED",
        extra={"false_success_injected_on": "add_member"},
    )
    return result
